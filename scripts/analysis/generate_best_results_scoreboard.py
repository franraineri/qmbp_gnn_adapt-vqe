#!/usr/bin/env python3
"""Generate Best Results Scoreboard — best-ever ΔE/gap per (topology × N).

Parses all evaluation reports in results/extrapolation_evals/ and for each
(topology, N) combination, finds the best result achieved at h≈2.5
(the hardest h-value near h_critical where the gap is smallest).

Output: one file per p — results/best_results_scoreboard_p{p}.md (+ .json)

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
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger("scoreboard")

EVAL_DIR = ROOT / "results" / "extrapolation_evals"
COMPARISON_DIR = ROOT / "results" / "model_comparison"
RESULTS_DIR = ROOT / "results"


def scoreboard_md_path(p_layers: int) -> Path:
    """Markdown scoreboard path for a given p (one file per p)."""
    return RESULTS_DIR / f"best_results_scoreboard_p{p_layers}.md"


def scoreboard_json_path(p_layers: int) -> Path:
    """JSON scoreboard path for a given p (one file per p)."""
    return RESULTS_DIR / f"best_results_scoreboard_p{p_layers}.json"


# Backward-compat: the legacy single-file paths default to the p=1 view so
# existing consumers that read best_results_scoreboard.json keep working.
OUTPUT_PATH = scoreboard_md_path(1)
OUTPUT_JSON_PATH = scoreboard_json_path(1)

TARGET_H: float = 2.5
H_TOLERANCE: float = 0.15  # Accept h within ±0.15 of target

# Dedicated chain_1d table near the critical point h_c=1.0
CRITICAL_TOPOLOGY: str = "chain_1d"
CRITICAL_H: float = 1.0
CRITICAL_H_TOLERANCE: float = 0.15  # captures h in [0.85, 1.15]


@dataclass
class PerHResult:
    """A single per-h measurement extracted from an eval report."""

    h: float
    e_pred: float
    e_exact: float
    abs_error: float
    gap: float
    de_gap: float
    fidelity: float | None = None
    fidelity_is_bound: bool = False


@dataclass
class EvalEntry:
    """One (topology, p, N) result from a single eval report."""

    topology: str
    n_qubits: int
    p_layers: int
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
    p_layers: int
    best_de_gap: float
    best_abs_error: float
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
    fidelity: float | None = None
    fidelity_is_bound: bool = False


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
    p_layers = 1

    # Infer topology and p from directory name: {topo}_p{N}/
    parent_dir = report_path.parent.name  # e.g. "chain_1d_p1"
    topo_match = re.match(r"(.+)_p(\d+)$", parent_dir)
    if topo_match:
        topology = topo_match.group(1)
        p_layers = int(topo_match.group(2))

    for line in lines[:20]:  # Header is in first 20 lines
        if line.startswith("**Model**:"):
            checkpoint = line.split(":", 1)[1].strip()
        elif line.startswith("**Date**:"):
            date_str = line.split(":", 1)[1].strip()
        elif line.startswith("**p_layers**:"):
            # Header p_layers (when present) is authoritative over the dir name.
            try:
                p_layers = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("**Multi-topology**:"):
            mt_val = line.split(":", 1)[1].strip().lower()
            is_mt = mt_val in ("yes", "true")
        elif "_MT_" in report_path.name:
            is_mt = True

    # Parse per-N sections
    current_n: int | None = None
    in_table = False
    col_idx: dict[str, int] = {}  # header-name → column index (format-robust)

    def _split_cells(row: str) -> list[str]:
        """Split a markdown table row into cells, robust to literal '|' inside
        column names like ``|ΔE|`` and ``|ΔE|/N``.
        """
        safe = row.replace("|ΔE|/N", "ABSERR_PER_SITE").replace("|ΔE|", "ABSERR")
        return [c.strip() for c in safe.split("|")[1:-1]]

    def _norm(col: str) -> str:
        """Normalize a table header cell to a canonical key."""
        c = col.strip().lower()
        if c in ("abserr", "abs_error"):
            return "abs_error"
        if c in ("abserr_per_site", "|δe|/n"):
            return "abs_error_per_site"
        if c in ("δe/gap", "de/gap", "de_gap"):
            return "de_gap"
        return c

    for line in lines:
        # Detect N section: "## N = 60 (119 params)"
        n_match = re.match(r"^## N\s*=\s*(\d+)", line)
        if n_match:
            current_n = int(n_match.group(1))
            in_table = False
            col_idx = {}
            continue

        # Detect table header and build a column-name → index map so parsing is
        # robust to column additions/reordering (e.g. the Fidelity column, or
        # the older |ΔE|/N layout). Never rely on fixed positional indices.
        if current_n is not None and "| h |" in line and "E_pred" in line:
            header_cells = _split_cells(line)
            col_idx = {_norm(c): i for i, c in enumerate(header_cells)}
            in_table = True
            continue

        # Skip separator
        if in_table and line.startswith("|---"):
            continue

        # Parse table row
        if in_table and current_n is not None and line.startswith("|"):
            parts = _split_cells(line)  # robust to literal '|' in headers
            if len(parts) < 6 or not col_idx:
                continue

            def _get(name: str) -> str | None:
                i = col_idx.get(name)
                return parts[i] if i is not None and i < len(parts) else None

            try:
                h_val = float(_get("h"))
            except (ValueError, TypeError):
                continue

            # Check if this h is close to target
            if abs(h_val - target_h) > H_TOLERANCE:
                continue

            try:
                e_pred = float(_get("e_pred"))
                e_exact = float(_get("e_exact"))
                abs_error = float(_get("abs_error"))
                gap = float(_get("gap"))
                de_gap = float(_get("de_gap"))
            except (ValueError, TypeError):
                continue

            # Fidelity (optional column; annotated with ≥ when a lower bound).
            fidelity: float | None = None
            fidelity_is_bound = False
            fid_raw = _get("fidelity")
            if fid_raw and fid_raw.upper() != "N/A":
                fidelity_is_bound = fid_raw.startswith("≥")
                try:
                    fidelity = float(fid_raw.lstrip("≥").strip())
                except ValueError:
                    fidelity = None

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
                if gap > 1e-10:
                    de_gap = abs_error / gap

            result = PerHResult(
                h=h_val,
                e_pred=e_pred,
                e_exact=e_exact,
                abs_error=abs_error,
                gap=gap,
                de_gap=de_gap,
                fidelity=fidelity,
                fidelity_is_bound=fidelity_is_bound,
            )

            rel_path = str(report_path.relative_to(ROOT))
            entries.append(
                EvalEntry(
                    topology=topology,
                    n_qubits=current_n,
                    p_layers=p_layers,
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


def scan_all_reports(
    target_h: float = TARGET_H,
    *,
    p_filter: int | None = None,
) -> list[EvalEntry]:
    """Scan all eval reports from extrapolation_evals/ AND model_comparison/.

    Both directories use the same markdown format (generated by
    generate_evaluation_report from analysis.evaluation_report).

    Parameters
    ----------
    target_h : float
        Reference h-value to extract per-report.
    p_filter : int | None
        If set, only keep entries with this p_layers value. None = all p.
    """
    all_entries: list[EvalEntry] = []
    n_files = 0
    n_parse_errors = 0

    for scan_dir in (EVAL_DIR, COMPARISON_DIR):
        if not scan_dir.exists():
            logger.debug("scan: dir does not exist, skipping: %s", scan_dir)
            continue

        for topo_dir in sorted(scan_dir.iterdir()):
            if not topo_dir.is_dir() or topo_dir.name.startswith("_"):
                continue

            for report_file in sorted(topo_dir.glob("eval_*.md")):
                n_files += 1
                try:
                    entries = parse_eval_report(report_file, target_h=target_h)
                    if p_filter is not None:
                        entries = [e for e in entries if e.p_layers == p_filter]
                    all_entries.extend(entries)
                except Exception as e:
                    n_parse_errors += 1
                    print(f"  ⚠️ Error parsing {report_file.name}: {e}", file=sys.stderr)
                    logger.warning("scan: parse error in %s: %s", report_file.name, e)

    logger.info(
        "scan: %d files → %d entries at h≈%.2f (p_filter=%s, %d parse errors)",
        n_files,
        len(all_entries),
        target_h,
        p_filter,
        n_parse_errors,
    )
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
        exp_dir / "exp_large_n_extrapolation",
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
) -> dict[str, dict[int, dict[int, BestResult]]]:
    """For each (topology, p, N), find the entry with the lowest |ΔE|.

    Returns nested dict: {topology: {p_layers: {n_qubits: BestResult}}}.
    Grouping by p_layers ensures p=1 and p=2 results never compete against
    each other (they are different ansätze — see p-layers-convention).
    """
    from qmbp_simulation.analysis.constants import compute_quality_score

    # Group by (topology, p, N)
    grouped: dict[tuple[str, int, int], list[EvalEntry]] = {}
    for entry in entries:
        key = (entry.topology, entry.p_layers, entry.n_qubits)
        grouped.setdefault(key, []).append(entry)

    logger.info(
        "compute_best: %d entries → %d (topology, p, N) groups",
        len(entries),
        len(grouped),
    )

    # Deduplicate: same (checkpoint, N, h, de_gap) from different directories
    # Keep the one with the lower ΔE/gap (they should be identical, but prefer
    # the entry from extrapolation_evals as canonical source if tied).
    for key in grouped:
        seen: dict[tuple[str, int, float], EvalEntry] = {}
        deduped = []
        for entry in grouped[key]:
            dedup_key = (entry.checkpoint, entry.n_qubits, round(entry.result.abs_error, 6))
            if dedup_key not in seen:
                seen[dedup_key] = entry
                deduped.append(entry)
            else:
                # Keep the one with lower |ΔE| (prefer extrapolation_evals source)
                existing = seen[dedup_key]
                if entry.result.abs_error < existing.result.abs_error:
                    deduped.remove(existing)
                    deduped.append(entry)
                    seen[dedup_key] = entry
        grouped[key] = deduped

    results: dict[str, dict[int, dict[int, BestResult]]] = {}

    for (topo, p_layers, n), group in sorted(grouped.items()):
        # Find entry with lowest |ΔE| (abs_error) — primary ranking criterion
        best_entry = min(group, key=lambda e: e.result.abs_error)
        r = best_entry.result

        # Grade based on |ΔE| (absolute energy error).
        # Thresholds are N-independent — what matters is the raw energy accuracy.
        #   A: |ΔE| < 0.05 (chemical accuracy regime)
        #   B: |ΔE| < 0.10
        #   C: |ΔE| < 0.30
        #   D: |ΔE| < 1.00
        #   F: |ΔE| ≥ 1.00
        if r.abs_error < 0.05:
            grade = "A"
        elif r.abs_error < 0.10:
            grade = "B"
        elif r.abs_error < 0.30:
            grade = "C"
        elif r.abs_error < 1.00:
            grade = "D"
        else:
            grade = "F"

        # Composite score for reference (higher = better)
        score = compute_quality_score(
            mean_de_gap=r.de_gap,
            p90_de_gap=r.de_gap,
            mean_abs_error_per_site=r.abs_error / max(n, 1),
            n_points=8,  # treat as full-confidence for scoreboard display
        )

        model_type = "MT" if best_entry.is_mt else "ST"

        logger.debug(
            "best[%s p%d N%d]: |ΔE|=%.4f grade=%s h=%.2f ckpt=%s (from %d candidates)",
            topo,
            p_layers,
            n,
            r.abs_error,
            grade,
            r.h,
            best_entry.checkpoint[:32],
            len(group),
        )

        # Find corresponding JSON for traceability
        run_json = _find_run_json(best_entry.report_file)

        best = BestResult(
            topology=topo,
            n_qubits=n,
            p_layers=p_layers,
            best_de_gap=r.de_gap,
            best_abs_error=r.abs_error,
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
            fidelity=r.fidelity,
            fidelity_is_bound=r.fidelity_is_bound,
        )

        results.setdefault(topo, {}).setdefault(p_layers, {})[n] = best

    return results


def _format_best_table(n_results: dict[int, BestResult]) -> list[str]:
    """Build the per-N markdown table lines for one (topology, p) group.

    Shared by the main scoreboard and the critical-h table to avoid
    duplicating the row-formatting logic.
    """
    lines: list[str] = []

    h_values_used = sorted({r.h_used for r in n_results.values()})
    if len(h_values_used) == 1:
        lines.append(f"**h used**: {h_values_used[0]:.3f}")
    else:
        lines.append(f"**h used**: varies ({min(h_values_used):.3f} – {max(h_values_used):.3f})")
    lines.append("")

    # ΔE/gap and gap columns temporarily hidden. Rollback: restore the
    # header/separator below and add `{r.best_de_gap:.4f} | {r.gap_at_best:.4f}`
    # to the row f-string.
    #   "| N | |ΔE| | ΔE/gap | gap | Fidelity | Grade | Model | Checkpoint | Date | Source |"
    #   "|--:|-----:|-------:|----:|:--------:|:-----:|:-----:|-----------|------|--------|"
    lines.append("| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |")
    lines.append("|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|")

    for n in sorted(n_results.keys()):
        r = n_results[n]
        ckpt_short = r.checkpoint
        if len(ckpt_short) > 40:
            ckpt_short = ckpt_short[:37] + "..."

        source_link = f"`{Path(r.report_file).name}`"
        if r.run_json:
            source_link += f" ([json]({r.run_json}))"

        if r.fidelity is None:
            fid_cell = "N/A"
        elif r.fidelity_is_bound:
            fid_cell = f"≥{r.fidelity:.4f}"
        else:
            fid_cell = f"{r.fidelity:.4f}"

        # Rollback: reinsert `{r.best_de_gap:.4f} | {r.gap_at_best:.4f} | `
        # after best_abs_error to bring back the ΔE/gap and gap columns.
        lines.append(
            f"| {n} | {r.best_abs_error:.4f} | "
            f"{fid_cell} | {r.grade} | {r.model_type} | "
            f"{ckpt_short} | {r.date[:10] if r.date else '—'} | "
            f"{source_link} |"
        )

    return lines


def format_markdown(
    best_for_p: dict[str, dict[int, BestResult]],
    target_h: float,
    n_reports_total: int,
    p_layers: int,
) -> str:
    """Format the scoreboard markdown for a SINGLE p value.

    Parameters
    ----------
    best_for_p : dict[str, dict[int, BestResult]]
        {topology: {n_qubits: BestResult}} for one p_layers.
    target_h : float
        Reference h-value.
    n_reports_total : int
        Number of unique reports scanned (for this p).
    p_layers : int
        The HVA depth this scoreboard covers.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Best Results Scoreboard — p={p_layers}",
        "",
        f"**Updated**: {now}",
        f"**p_layers**: {p_layers}",
        f"**Reference h-value**: {target_h:.2f} "
        "(hardest region near h_critical; actual h used noted per entry)",
        f"**Reports scanned**: {n_reports_total}",
        f"**Criterion**: Best ΔE/gap achieved at h≈{target_h:.1f} per (topology × N)",
        "",
        f"> This report shows the **best single-point result ever achieved** at h≈{target_h:.1f} "
        f"for each (topology, N) combination at **p={p_layers}**, in the "
        "**extrapolation regime** (N values tested with MPNN zero-shot prediction).",
        "> Each p has its own scoreboard file — p=1 and p=2 are different ansätze "
        "and never compete against each other.",
        "> It does NOT average over h — it tracks the hardest operating point near h_critical.",
        "> For in-distribution quality (training N), see `model_evaluation_report.md`.",
        "> Grade thresholds: A (|ΔE|<0.05), B (<0.10), C (<0.30), D (<1.00), F (≥1.00).",
        "",
        "---",
        "",
    ]

    # ── Global summary table (one row per topology) ──────────────────────
    lines.append("## Summary: Best Grade per Topology")
    lines.append("")
    lines.append(
        "| Topology | Max N evaluated | Best grade | "
        "Best |ΔE| (any N) | Best model type | N trained up to |"
    )
    lines.append("|---|---|---|---|---|---|")

    for topo in sorted(best_for_p.keys()):
        n_results = best_for_p[topo]
        if not n_results:
            continue
        max_n = max(n_results.keys())
        best_grade_entry = min(n_results.values(), key=lambda r: r.best_abs_error)
        reasonable_ns = [n for n, r in n_results.items() if r.best_abs_error < 10.0]
        max_n_trained = max(reasonable_ns) if reasonable_ns else max_n

        lines.append(
            f"| {topo} | {max_n} | {best_grade_entry.grade} | "
            f"{best_grade_entry.best_abs_error:.4f} | "
            f"{best_grade_entry.model_type} | {max_n_trained} |"
        )

    lines.extend(["", "---", ""])

    # ── Per-topology detailed tables ─────────────────────────────────────
    for topo in sorted(best_for_p.keys()):
        n_results = best_for_p[topo]
        if not n_results:
            continue
        lines.append(f"## {topo}")
        lines.append("")
        lines.extend(_format_best_table(n_results))
        lines.append("")
        lines.append("")

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


def format_critical_h_table(
    p_layers: int,
    topology: str = CRITICAL_TOPOLOGY,
    target_h: float = CRITICAL_H,
    tolerance: float = CRITICAL_H_TOLERANCE,
) -> str:
    """Build a dedicated best-result table for one topology near h_critical, for one p.

    Reuses the same scan + ranking machinery as the main scoreboard
    (scan_all_reports + compute_best_per_topology_n), just retargeted to a
    different h. By default: chain_1d near h=1.0 (the critical point, where the
    gap collapses and the ansatz is most stressed).

    Returns the markdown section (empty string if no entries found).
    """
    # Reuse scan with the critical target-h and a scoped tolerance, filtered by p.
    global H_TOLERANCE
    saved_tol = H_TOLERANCE
    H_TOLERANCE = tolerance
    try:
        entries = [
            e
            for e in scan_all_reports(target_h=target_h, p_filter=p_layers)
            if e.topology == topology
        ]
    finally:
        H_TOLERANCE = saved_tol

    if not entries:
        return (
            f"## {topology} near h_critical (h≈{target_h:.1f})\n\n"
            f"_No p={p_layers} eval reports found with h in "
            f"[{target_h - tolerance:.2f}, {target_h + tolerance:.2f}]._\n"
        )

    best_by_topo = compute_best_per_topology_n(entries)
    # {p: {n: BestResult}} for this topology; we filtered to a single p above.
    p_results = best_by_topo.get(topology, {})
    n_results = p_results.get(p_layers, {})

    lines = [
        f"## {topology} near h_critical (h≈{target_h:.1f})",
        "",
        f"Best-ever single-point result for **{topology}** at h≈{target_h:.1f} "
        f"(±{tolerance:.2f}), the critical region where the gap is smallest and the "
        "ansatz is most stressed. Same ranking as the main scoreboard "
        "(lowest |ΔE| per N).",
        "",
    ]
    if n_results:
        lines.extend(_format_best_table(n_results))
        lines.append("")

    return "\n".join(lines)


def cross_validate_with_registry(
    best_by_topo: dict[str, dict[int, dict[int, BestResult]]],
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
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        db = ModelRegistryDB()
        manifest = _load_manifest()

        # Build a map: checkpoint_file → zoo entry
        zoo_map = {e.checkpoint_file: e for e in manifest}

        # Flatten {topo: {p: {n: BestResult}}} → iterate every BestResult
        flat = [
            best
            for p_results in best_by_topo.values()
            for n_results in p_results.values()
            for best in n_results.values()
        ]
        for best in sorted(flat, key=lambda b: (b.topology, b.p_layers, b.n_qubits)):
            topo = best.topology
            n = best.n_qubits
            p = best.p_layers
            if True:
                ckpt_name = Path(best.checkpoint).name
                if ckpt_name not in zoo_map:
                    # Try partial match, preferring a zoo entry with the SAME p
                    # so we never cross-check a p=1 scoreboard row against a
                    # p=2 zoo entry (their pass_rate_by_n are incomparable).
                    matches = [k for k in zoo_map if ckpt_name.startswith(k[:20])]
                    p_matches = [k for k in matches if zoo_map[k].p_layers == p]
                    matches = p_matches or matches
                    if not matches:
                        continue
                    ckpt_name = matches[0]

                zoo_entry = zoo_map[ckpt_name]

                # p-coherence guard: only cross-check zoo entries at the SAME p.
                # A p mismatch means the resolved zoo entry is a different ansatz
                # depth — its pass_rate_by_n cannot be compared to this row.
                if zoo_entry.p_layers != p:
                    logger.debug(
                        "cross_val: skip %s (%s N=%d p=%d): zoo entry p=%d mismatch",
                        ckpt_name,
                        topo,
                        n,
                        p,
                        zoo_entry.p_layers,
                    )
                    continue

                # Check 1: pass_rate_by_n consistency
                if zoo_entry.pass_rate_by_n:
                    n_str = str(n)
                    if n_str in zoo_entry.pass_rate_by_n:
                        zoo_pass_at_n = float(zoo_entry.pass_rate_by_n[n_str])
                        # If zoo says 0% pass but we show grade A/B, flag discrepancy
                        if best.grade in ("A", "B") and zoo_pass_at_n < 0.20:
                            validation_lines.append(
                                f"⚠️ {topo} p={p} N={n}: scoreboard grade={best.grade} but "
                                f"zoo pass_rate_by_n[{n}]={zoo_pass_at_n:.0%} — "
                                f"possible stale zoo data"
                            )
                        # If zoo says good pass but we show F, flag
                        elif best.grade == "F" and zoo_pass_at_n > 0.50:
                            validation_lines.append(
                                f"⚠️ {topo} p={p} N={n}: scoreboard grade=F (|ΔE|={best.best_abs_error:.3f}) "
                                f"but zoo pass_rate_by_n[{n}]={zoo_pass_at_n:.0%} — "
                                f"possible inflated zoo or h-grid difference"
                            )

                # Check 2: Registry evaluation records
                record = db.get_model(ckpt_name)
                if record and record.evaluations:
                    latest_eval = record.evaluations[-1]
                    # If latest eval target_n includes this N, cross-check
                    if n in (latest_eval.target_n_values or []):
                        if (
                            latest_eval.mean_de_gap > 0
                            and abs(best.best_de_gap - latest_eval.mean_de_gap)
                            / max(latest_eval.mean_de_gap, 1e-8)
                            > 1.0
                        ):
                            # Our per-h best is very different from the registry's mean
                            # This is expected (we pick best h, registry averages all h)
                            # Only flag if direction is inverted
                            if best.best_abs_error > latest_eval.mean_de_gap * best.gap_at_best * 3:
                                validation_lines.append(
                                    f"⚠️ {topo} p={p} N={n}: scoreboard |ΔE|@h=2.5 = {best.best_abs_error:.3f} "
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
    p_filter: int | None = None,
) -> dict:
    """Main entry point: scan reports, compute best, write markdown.

    Parameters
    ----------
    target_h : float
        Reference h-value to track.
    output_json : bool
        Also write JSON output.
    p_filter : int | None
        If set, only include results for this p_layers. None = all p.

    Returns summary dict for programmatic use.
    """
    entries = scan_all_reports(target_h=target_h, p_filter=p_filter)

    if not entries:
        _msg = (
            f"  ⚠️ No evaluation reports found for p={p_filter}"
            if p_filter is not None
            else "  ⚠️ No evaluation reports found in results/extrapolation_evals/"
        )
        print(_msg)
        return {
            "n_entries_parsed": 0,
            "n_reports_scanned": 0,
            "best_by_topology": {},
            "best_by_p": {},
            "files_written": [],
        }

    best_by_topo = compute_best_per_topology_n(entries)

    # ── GT cache freshness check (validate e_exact against authoritative source) ──
    gt_warnings: list[str] = []
    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        for topo, p_results in best_by_topo.items():
            for _p, n_results in p_results.items():
                for n, best in n_results.items():
                    cached = gt_cache.get(topo, n, "tfim_bond_resolved", best.h_used)
                    if cached is None:
                        cached = gt_cache.get(topo, n, "tfim", best.h_used)
                    if cached is not None:
                        gt_energy = float(cached["energy"])
                        if abs(gt_energy - best.e_exact) > 0.001:
                            gt_warnings.append(
                                f"⚠️ {topo} p={_p} N={n} h={best.h_used:.2f}: "
                                f"report e_exact={best.e_exact:.4f} vs GT cache={gt_energy:.4f} "
                                f"(Δ={abs(gt_energy - best.e_exact):.4f}) — STALE e_exact in report"
                            )
    except Exception:
        pass  # GT cache validation is best-effort

    if gt_warnings:
        print(f"  ⚠️ GT freshness issues ({len(gt_warnings)}):")
        for w in gt_warnings[:5]:
            print(f"    {w}")

    # Count unique reports
    unique_reports = set(e.report_file for e in entries)
    n_reports = len(unique_reports)

    # Which p values to emit a scoreboard for.
    p_values = sorted({e.p_layers for e in entries}) if p_filter is None else [p_filter]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target_h": target_h,
        "n_reports_scanned": n_reports,
        "n_entries_parsed": len(entries),
        "p_values": p_values,
        "files_written": [],
        # Per-p best results: {"p1": {topo: {n_str: entry}}, ...}
        "best_by_p": {},
        # Backward-compat alias populated with the LOWEST p present, so existing
        # consumers reading best_by_topology (metrics.py, validate_scoreboard.py,
        # generate_thesis_tables.py) keep working.
        "best_by_topology": {},
    }

    # ── Emit one scoreboard file per p ───────────────────────────────────
    for p in p_values:
        best_for_p = {
            topo: p_results[p]
            for topo, p_results in best_by_topo.items()
            if p in p_results and p_results[p]
        }
        if not best_for_p:
            continue

        # Markdown for this p (reuses format_markdown + registry cross-val)
        md_content = format_markdown(best_for_p, target_h, n_reports, p_layers=p)

        # Cross-validate with ModelRegistryDB (wrap in the 3-level shape it expects)
        validation_warnings = cross_validate_with_registry(
            {topo: {p: n_results} for topo, n_results in best_for_p.items()}
        )
        if validation_warnings:
            validation_section = ["", "---", "", "## Cross-Validation (vs ModelRegistryDB)", ""]
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

        # Dedicated chain_1d table near the critical point (this p only).
        critical_table_h1 = format_critical_h_table(p_layers=p, target_h=1.0)
        if critical_table_h1:
            md_content += "\n" + "\n".join(["", "---", "", critical_table_h1])

        critical_table_h1_5 = format_critical_h_table(p_layers=p, target_h=1.5)
        if critical_table_h1_5:
            md_content += "\n" + "\n".join(["", "---", "", critical_table_h1_5])

        md_path = scoreboard_md_path(p)
        md_path.write_text(md_content)
        summary["files_written"].append(str(md_path.relative_to(ROOT)))
        print(f"  📊 Best Results Scoreboard (p={p}): {md_path.relative_to(ROOT)}")

        # Per-p JSON view
        p_json = {
            str(topo): {str(n): asdict(r) for n, r in sorted(nr.items())}
            for topo, nr in sorted(best_for_p.items())
        }
        summary["best_by_p"][f"p{p}"] = p_json

        if output_json:
            per_p_summary = {
                "generated_at": summary["generated_at"],
                "target_h": target_h,
                "p_layers": p,
                "n_reports_scanned": n_reports,
                "best_by_topology": p_json,
            }
            json_path = scoreboard_json_path(p)
            json_path.write_text(json.dumps(per_p_summary, indent=2, default=str))
            print(f"  📄 JSON output (p={p}): {json_path.relative_to(ROOT)}")

    # Legacy alias: lowest p present (usually p=1) for backward compatibility.
    if p_values:
        legacy_p_key = f"p{p_values[0]}"
        summary["best_by_topology"] = summary["best_by_p"].get(legacy_p_key, {})

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
    parser.add_argument(
        "--p-layers",
        type=int,
        default=None,
        help="If set, only include results for this p_layers (e.g. 1 or 2). "
        "Default: all p (each shown in its own section).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity: -v = INFO, -vv = DEBUG (per-group "
        "best-selection traces).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global H_TOLERANCE
    H_TOLERANCE = args.tolerance

    _level = logging.WARNING
    if args.verbose == 1:
        _level = logging.INFO
    elif args.verbose >= 2:
        _level = logging.DEBUG
    logging.basicConfig(level=_level, format="%(levelname)s [%(name)s] %(message)s")
    logger.setLevel(_level)

    summary = generate_scoreboard(
        target_h=args.target_h, output_json=args.json, p_filter=args.p_layers
    )

    if summary["n_entries_parsed"] == 0:
        return 1

    # Print quick summary
    print(
        f"\n  Scanned {summary['n_reports_scanned']} reports, "
        f"found {summary['n_entries_parsed']} entries at h≈{args.target_h}"
    )

    # best_by_p = {"p{p}": {topo: {n_str: result_dict}}}
    by_p = summary.get("best_by_p", {})
    for p_key in sorted(by_p.keys()):
        for topo, n_results in sorted(by_p[p_key].items()):
            if not n_results:
                continue
            n_vals = sorted(int(k) for k in n_results.keys())
            best_grade = min(n_results.values(), key=lambda r: r["best_abs_error"])
            print(
                f"  {topo:14s} {p_key}: N={n_vals}, "
                f"best |ΔE|={best_grade['best_abs_error']:.4f} ({best_grade['grade']})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
