#!/usr/bin/env python3
"""Auto-update accelerated_cross_n_coverage.md from dashboard + NPZ data.

Reads the model quality dashboard (auto-generated after every run) and
the NPZ training data to regenerate the data-driven sections of the
coverage document. Preserves hand-written sections (thesis framing,
root cause analysis, recommendations) unchanged.

Usage:
    .venv/bin/python scripts/maintenance/update_cross_n_coverage.py
    .venv/bin/python scripts/maintenance/update_cross_n_coverage.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COVERAGE_DOC = ROOT / "internal" / "documentation" / "analysis" / "accelerated_cross_n_coverage.md"
DASHBOARD_PATH = DATA / "model_quality_dashboard.json"
NPZ_DIR = DATA / "multi_n_training"


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dashboard() -> dict:
    if not DASHBOARD_PATH.exists():
        print("  ⚠️  Dashboard not found. Run any experiment first.", file=sys.stderr)
        sys.exit(1)
    with open(DASHBOARD_PATH) as f:
        return json.load(f)


def compute_gt_coverage(npz_dir: Path) -> dict[str, int]:
    """Count h-points in NPZ that have no GT cache entry. Returns {filename: n_missing}."""
    gt_path = DATA / "ground_truth_cache.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        raw = json.load(f)
    gt_entries = raw.get("entries", raw) if isinstance(raw, dict) else {}

    missing = {}
    for npz_file in sorted(npz_dir.glob("*.npz")):
        d = np.load(str(npz_file), allow_pickle=True)
        h_vals = d["h_values"]
        parts = npz_file.stem.split("_")
        n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
        if n_idx is None:
            continue
        topo = "_".join(parts[:n_idx])
        n_val = int(parts[n_idx][1:])
        n_miss = sum(
            1 for h in h_vals
            if f"{topo}|{n_val}|tfim_bond_resolved|{float(h):.6f}" not in gt_entries
        )
        if n_miss > 0:
            missing[npz_file.name] = n_miss
    return missing


def count_zoo_orphans() -> int:
    manifest_path = DATA / "model_zoo" / "manifest.json"
    ckpt_dir = DATA / "model_zoo" / "checkpoints"
    if not manifest_path.exists() or not ckpt_dir.exists():
        return 0
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    registered = {e.get("checkpoint_file") for e in entries}
    return sum(1 for f in ckpt_dir.glob("*.pt") if f.name not in registered)


# ═══════════════════════════════════════════════════════════════════════════════
# Section generators
# ═══════════════════════════════════════════════════════════════════════════════

def generate_executive_summary(dashboard: dict) -> str:
    topo_sum = dashboard.get("topology_summary", {})
    configs = dashboard.get("configs", [])

    rows = []
    for topo in sorted(topo_sum.keys()):
        info = topo_sum[topo]
        topo_configs = [c for c in configs if c["topology"] == topo]
        total_pts = sum(c["n_points"] for c in topo_configs)
        n_values = info.get("n_values", [])
        h_min = min(c["h_range"][0] for c in topo_configs if c.get("h_range"))
        h_max = max(c["h_range"][1] for c in topo_configs if c.get("h_range"))

        # Zoo multi-N pass rate
        zoo_multi = next(
            (c["zoo_pass_rate"] for c in topo_configs if c.get("n_qubits") == 0 and c.get("zoo_pass_rate")),
            None
        )
        zoo_str = f"{zoo_multi:.0%}" if zoo_multi else "—"

        n_max_viable = info.get("n_max_viable", "—")
        best_pass = info.get("best_pass_rate_5pct", 0)

        rows.append(
            f"| {topo} | {','.join(str(n) for n in n_values)} | "
            f"{total_pts} | [{h_min:.1f}, {h_max:.1f}] | "
            f"{best_pass:.0%} | {zoo_str} | {n_max_viable} |"
        )

    lines = [
        "## Resumen Ejecutivo (auto-generated)",
        "",
        "| Topología | N values | Total pts | h-range | Best pass@5% | Zoo (multi-N) | n_max_viable |",
        "|-----------|---------|-----------|---------|--------------|---------------|--------------|",
    ] + rows
    return "\n".join(lines)


def generate_topology_table(topo: str, configs: list[dict]) -> str:
    """Generate the per-topology detail table from dashboard data."""
    topo_configs = sorted(
        [c for c in configs if c["topology"] == topo],
        key=lambda c: c["n_qubits"]
    )
    if not topo_configs:
        return f"*No data for {topo}*"

    rows = []
    for c in topo_configs:
        N = c["n_qubits"]
        pts = c["n_points"]
        h_range = c.get("h_range", [0, 0])
        h_str = f"[{h_range[0]:.2f}, {h_range[1]:.2f}]" if len(h_range) == 2 else "—"
        pass5 = c.get("pass_rate_5pct", 0)
        pass_dual = c.get("pass_rate_dual_criterion", pass5)
        h_front = c.get("h_frontier")
        h_front_str = f"{h_front:.2f}" if h_front is not None else "N/A"
        smooth = c.get("theta_smoothness")
        smooth_str = f"{smooth:.2f} {'⚠️' if smooth and smooth > 0.5 else ''}" if smooth is not None else "—"

        # Gap masking detection
        gap_mask = pass5 - pass_dual
        mask_str = ""
        if gap_mask > 0.15:
            mask_str = f" ⚠️ GAP MASK +{gap_mask:.0%}"
        elif gap_mask > 0.05:
            mask_str = f" ({gap_mask:.0%} masked)"

        zoo_div = c.get("zoo_vs_npz_divergence")
        div_str = f"div={zoo_div:.2f}" if zoo_div is not None else ""
        stale = " STALE" if c.get("model_stale") else ""

        obs = f"{mask_str} {div_str}{stale}".strip() or "—"

        rows.append(
            f"| {N} | {pts} | {h_str} | {pass5:.0%} | {pass_dual:.0%} | {h_front_str} | {smooth_str} | {obs} |"
        )

    lines = [
        f"### {topo.replace('_', ' ').title()}",
        "",
        "| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |",
        "|---|--------|---------|---------|-----------|------------|---------|-------------|",
    ] + rows
    return "\n".join(lines)


def generate_gap_masking_table(configs: list[dict]) -> str:
    """Show configs where pass_rate_5pct >> pass_rate_dual (gap masking)."""
    masked = [
        c for c in configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > 0.10
    ]
    if not masked:
        return "*(No significant gap masking detected)*"

    rows = []
    for c in sorted(masked, key=lambda x: -(x["pass_rate_5pct"] - x.get("pass_rate_dual_criterion", 0))):
        diff = c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0)
        rows.append(
            f"| {c['topology']} | {c['n_qubits']} | "
            f"{c['pass_rate_5pct']:.0%} | {c.get('pass_rate_dual_criterion', 0):.0%} | "
            f"{diff:.0%} |"
        )

    lines = [
        "| Topology | N | Pass@5% | Pass@dual | Gap masked |",
        "|----------|---|---------|-----------|------------|",
    ] + rows
    return "\n".join(lines)


def generate_h_frontier_table(configs: list[dict]) -> str:
    """Cross-topology h_frontier table."""
    # Collect all topologies and N values
    by_topo_n = defaultdict(dict)
    for c in configs:
        if c.get("h_frontier") is not None:
            by_topo_n[c["topology"]][c["n_qubits"]] = c["h_frontier"]

    all_n = sorted(set(n for d in by_topo_n.values() for n in d.keys()))
    n_header = " | ".join(f"N={n}" for n in all_n)
    separator = " | ".join("---" for _ in all_n)

    rows = []
    for topo in sorted(by_topo_n.keys()):
        vals = []
        for n in all_n:
            v = by_topo_n[topo].get(n)
            vals.append(f"{v:.2f}" if v is not None else "—")
        rows.append(f"| {topo} | {' | '.join(vals)} |")

    lines = [
        f"| Topología | {n_header} |",
        f"|-----------|{separator}|",
    ] + rows
    return "\n".join(lines)


def generate_training_health_table(dashboard: dict, gt_missing: dict, n_orphans: int) -> str:
    """Training data health summary."""
    integrity = dashboard.get("integrity", {})
    configs = dashboard.get("configs", [])
    total_pts = sum(c.get("n_points", 0) for c in configs)
    n_nan = integrity.get("n_configs_with_nan_theta", 0)
    zoo_ok = integrity.get("zoo_integrity_ok", None)
    zoo_miss = integrity.get("zoo_n_missing", 0)

    n_stale = sum(1 for c in configs if c.get("model_stale"))
    n_retrain = sum(1 for c in configs if c.get("needs_retrain"))
    n_high_smooth = sum(1 for c in configs if c.get("theta_smoothness") and c["theta_smoothness"] > 0.5)
    total_missing_gt = sum(gt_missing.values())
    n_masked = sum(
        1 for c in configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > 0.10
    )

    rows = [
        f"| Total NPZ files | {len(configs)} | |",
        f"| Total training points | {total_pts} | |",
        f"| NaN in θ | {n_nan} configs | {'✅' if n_nan == 0 else '❌'} |",
        f"| Zoo integrity | {zoo_ok} | {'✅' if zoo_ok else '❌'} |",
        f"| Zoo missing | {zoo_miss} | {'✅' if zoo_miss == 0 else '⚠️'} |",
        f"| Zoo orphan checkpoints | {n_orphans} | {'⚠️ cleanup needed' if n_orphans > 0 else '✅'} |",
        f"| GT coverage gaps | {total_missing_gt} uncovered h-points | {'⚠️' if total_missing_gt > 0 else '✅'} |",
        f"| Stale zoo models | {n_stale} | {'⚠️' if n_stale > 0 else '✅'} |",
        f"| Need retrain | {n_retrain} | {'🔄' if n_retrain > 0 else '✅'} |",
        f"| High θ discontinuity (>0.5) | {n_high_smooth} configs | {'⚠️' if n_high_smooth > 0 else '✅'} |",
        f"| Gap masking detected | {n_masked} configs | {'⚠️' if n_masked > 0 else '✅'} |",
    ]

    lines = [
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
    ] + rows
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown document update
# ═══════════════════════════════════════════════════════════════════════════════

# Sentinel markers to locate auto-generated sections
BEGIN_MARKER = "<!-- AUTO-GENERATED-BEGIN:{section} -->"
END_MARKER = "<!-- AUTO-GENERATED-END:{section} -->"


def update_section(doc: str, section_id: str, new_content: str) -> tuple[str, bool]:
    """Replace content between sentinel markers. Returns (updated_doc, was_found)."""
    begin = BEGIN_MARKER.format(section=section_id)
    end = END_MARKER.format(section=section_id)

    if begin not in doc:
        return doc, False

    before = doc[:doc.index(begin) + len(begin)]
    after = doc[doc.index(end):]
    return before + "\n" + new_content + "\n" + after, True


def add_section_sentinels(doc: str, section_id: str, heading: str) -> str:
    """Add sentinel markers around an existing section heading."""
    begin = BEGIN_MARKER.format(section=section_id)
    end = END_MARKER.format(section=section_id)

    if begin in doc:
        return doc  # already has sentinels

    # Find the section heading
    idx = doc.find(f"\n{heading}")
    if idx == -1:
        idx = doc.find(f"\n## {heading}")
    if idx == -1:
        return doc  # heading not found

    # Find the end of this section (next ## heading or end of doc)
    next_section = doc.find("\n## ", idx + 1)
    if next_section == -1:
        next_section = len(doc)

    section_content = doc[idx + 1:next_section]
    return (
        doc[:idx + 1] +
        begin + "\n" +
        section_content +
        end + "\n" +
        doc[next_section:]
    )


def generate_document(dashboard: dict, gt_missing: dict, n_orphans: int) -> str:
    """Generate the full updated document content."""
    configs = dashboard.get("configs", [])
    topo_sum = dashboard.get("topology_summary", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    topologies = sorted(topo_sum.keys())

    lines = [
        "# Accelerated Cross-N Coverage Analysis",
        "",
        f"**Fecha**: {now} (auto-generated by update_cross_n_coverage.py)",
        "**Modelo**: TFIM bond-resolved, p=1",
        "**Método**: AcceleratedVQE + UnifiedMPNN cross-N transfer",
        "**Fuentes**: `model_quality_dashboard.json`, NPZ training data, GT cache",
        "",
        "> **Nota**: Las secciones marcadas con `<!-- AUTO-GENERATED -->` se actualizan",
        "> automáticamente. Las secciones de análisis físico y recomendaciones son manuales.",
        "",
        "---",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:executive_summary -->",
        generate_executive_summary(dashboard),
        f"<!-- AUTO-GENERATED-END:executive_summary -->",
        "",
        "---",
        "",
        "## Training Data Health",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:health -->",
        generate_training_health_table(dashboard, gt_missing, n_orphans),
        f"<!-- AUTO-GENERATED-END:health -->",
        "",
        "---",
        "",
        "## Gap Masking Analysis",
        "",
        "Configs where `pass@5% - pass@dual_criterion > 10%` — large gap inflates ΔE/gap metric:",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:gap_masking -->",
        generate_gap_masking_table(configs),
        f"<!-- AUTO-GENERATED-END:gap_masking -->",
        "",
        "---",
        "",
        "## Detalle por Topología",
        "",
    ]

    for topo in topologies:
        lines.append(f"<!-- AUTO-GENERATED-BEGIN:topo_{topo} -->")
        lines.append(generate_topology_table(topo, configs))
        lines.append(f"<!-- AUTO-GENERATED-END:topo_{topo} -->")
        lines.append("")

    lines += [
        "---",
        "",
        "## h_frontier per Topology",
        "",
        "h_frontier = h below which ΔE/gap ≥ 5% (pipeline fails):",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:h_frontier -->",
        generate_h_frontier_table(configs),
        f"<!-- AUTO-GENERATED-END:h_frontier -->",
        "",
        "---",
        "",
        "## Cross-N Transfer Summary",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:cross_n_transfer -->",
        _generate_cross_n_transfer_table(configs, topo_sum),
        f"<!-- AUTO-GENERATED-END:cross_n_transfer -->",
        "",
        "---",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:training_plan -->",
        generate_training_plan(dashboard),
        f"<!-- AUTO-GENERATED-END:training_plan -->",
        "",
    ]

    return "\n".join(lines)


def _generate_cross_n_transfer_table(configs: list[dict], topo_sum: dict) -> str:
    rows = []
    for topo in sorted(topo_sum.keys()):
        info = topo_sum[topo]
        n_max = info.get("n_max_viable", "—")
        best_src = info.get("cross_n_best_source_for_largest")
        best_src_str = (
            f"train_n={best_src['train_n']} (@10%={best_src['pass_rate_10pct']:.0%})"
            if best_src else "no data"
        )
        rows.append(f"| {topo} | {n_max} | {info.get('best_pass_rate_5pct', 0):.0%} | {best_src_str} |")

    lines = [
        "| Topology | n_max_viable | Best pass@5% | Best cross-N source |",
        "|----------|-------------|-------------|---------------------|",
    ] + rows
    return "\n".join(lines)


def generate_training_plan(dashboard: dict) -> str:
    """Generate an actionable training plan from training_utility classification.

    Partitions configs into: DELETE (not_useful), IMPROVE (insufficient), EXPAND (useful).
    Provides specific commands for each action.
    """
    configs = dashboard.get("configs", [])

    not_useful = [c for c in configs if c.get("training_utility") == "not_useful"]
    insufficient = [c for c in configs if c.get("training_utility") == "insufficient_signal"]
    useful = [c for c in configs if c.get("training_utility") == "useful"]

    lines = ["## Training Plan (auto-generated)", ""]

    # Summary counts
    lines.append(f"**Total configs**: {len(configs)} | "
                 f"✅ Useful: {len(useful)} | "
                 f"⚠️ Insufficient: {len(insufficient)} | "
                 f"❌ Not useful: {len(not_useful)}")
    lines.append("")

    # DELETE section
    if not_useful:
        lines.append("### ❌ DELETE — Not useful for MPNN training")
        lines.append("")
        lines.append("These NPZ files teach the MPNN wrong mappings. Remove or regenerate:")
        lines.append("")
        lines.append("| File | Topology | N | Reason |")
        lines.append("|------|----------|---|--------|")
        for c in sorted(not_useful, key=lambda x: (x["topology"], x["n_qubits"])):
            reason = c.get("training_utility_reason", "")[:60]
            lines.append(f"| `{c['file']}` | {c['topology']} | {c['n_qubits']} | {reason} |")
        lines.append("")
        lines.append("```bash")
        for c in sorted(not_useful, key=lambda x: (x["topology"], x["n_qubits"])):
            lines.append(f"rm data/multi_n_training/{c['file']}")
        lines.append("```")
        lines.append("")

    # IMPROVE section
    if insufficient:
        lines.append("### ⚠️ IMPROVE — Insufficient signal (need more good points)")
        lines.append("")
        lines.append("Run iterative-improve to densify these configs above the frontier:")
        lines.append("")
        lines.append("| File | Topology | N | Pts | Dual pass | h_frontier | Action |")
        lines.append("|------|----------|---|-----|-----------|------------|--------|")
        for c in sorted(insufficient, key=lambda x: (x["topology"], x["n_qubits"])):
            hf = c.get("h_frontier")
            hf_str = f"{hf:.2f}" if hf else "N/A"
            dual = c.get("pass_rate_dual_criterion", 0)
            h_min_suggest = f"{hf + 0.2:.1f}" if hf else "h_frontier+0.2"
            action = f"iterative-improve h≥{h_min_suggest}"
            lines.append(
                f"| `{c['file']}` | {c['topology']} | {c['n_qubits']} | "
                f"{c['n_points']} | {dual:.0%} | {hf_str} | {action} |"
            )
        lines.append("")

    # EXPAND section
    if useful:
        lines.append("### ✅ EXPAND — Useful configs (add more h-points for better generalization)")
        lines.append("")
        lines.append("| Topology | N | Pts | Dual pass | h_frontier | Priority |")
        lines.append("|----------|---|-----|-----------|------------|----------|")
        for c in sorted(useful, key=lambda x: (x["n_qubits"], x["topology"])):
            hf = c.get("h_frontier")
            hf_str = f"{hf:.2f}" if hf else "N/A"
            dual = c.get("pass_rate_dual_criterion", 0)
            # Priority: configs with fewer points but good pass rate → densify
            if c["n_points"] < 20 and dual > 0.7:
                priority = "HIGH (few pts, good quality)"
            elif c["n_points"] < 30:
                priority = "MEDIUM (expand range)"
            else:
                priority = "LOW (already dense)"
            lines.append(
                f"| {c['topology']} | {c['n_qubits']} | "
                f"{c['n_points']} | {dual:.0%} | {hf_str} | {priority} |"
            )
        lines.append("")

    return "\n".join(lines)


def update_existing_document(existing: str, dashboard: dict, gt_missing: dict, n_orphans: int) -> str:
    """Update only the AUTO-GENERATED sections in an existing document."""
    configs = dashboard.get("configs", [])
    topo_sum = dashboard.get("topology_summary", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Update header date
    import re
    existing = re.sub(
        r"\*\*Fecha\*\*:.*",
        f"**Fecha**: {now} (auto-updated by update_cross_n_coverage.py)",
        existing,
        count=1,
    )

    # Update each auto-generated section
    updated, found = update_section(existing, "executive_summary", generate_executive_summary(dashboard))
    if found:
        existing = updated

    updated, found = update_section(existing, "health", generate_training_health_table(dashboard, gt_missing, n_orphans))
    if found:
        existing = updated

    updated, found = update_section(existing, "gap_masking", generate_gap_masking_table(configs))
    if found:
        existing = updated

    updated, found = update_section(existing, "h_frontier", generate_h_frontier_table(configs))
    if found:
        existing = updated

    updated, found = update_section(existing, "cross_n_transfer", _generate_cross_n_transfer_table(configs, topo_sum))
    if found:
        existing = updated

    updated, found = update_section(existing, "training_plan", generate_training_plan(dashboard))
    if found:
        existing = updated

    for topo in sorted(topo_sum.keys()):
        updated, found = update_section(existing, f"topo_{topo}", generate_topology_table(topo, configs))
        if found:
            existing = updated

    return existing


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Update accelerated_cross_n_coverage.md")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing")
    parser.add_argument(
        "--force-regenerate", action="store_true",
        help="Regenerate full document (discards manual sections). Default: update in-place.",
    )
    args = parser.parse_args()

    print("Loading dashboard...")
    dashboard = load_dashboard()
    n_configs = dashboard.get("n_configs", 0)
    generated_at = dashboard.get("generated_at", "unknown")
    print(f"  {n_configs} configs, generated at {generated_at[:19]}")

    print("Computing GT coverage gaps...")
    gt_missing = compute_gt_coverage(NPZ_DIR) if NPZ_DIR.exists() else {}
    if gt_missing:
        print(f"  ⚠️  {len(gt_missing)} NPZ files with uncovered GT h-points")

    print("Counting zoo orphans...")
    n_orphans = count_zoo_orphans()
    if n_orphans:
        print(f"  ⚠️  {n_orphans} orphan checkpoints")

    # Determine update strategy
    if args.force_regenerate or not COVERAGE_DOC.exists():
        print("Generating full document...")
        new_content = generate_document(dashboard, gt_missing, n_orphans)
        action = "create"
    else:
        print(f"Updating existing document: {COVERAGE_DOC.relative_to(ROOT)}")
        existing = COVERAGE_DOC.read_text()

        # Check if document has sentinel markers
        if "<!-- AUTO-GENERATED-BEGIN:" not in existing:
            print(
                "  ⚠️  Document has no AUTO-GENERATED markers.\n"
                "     The document will be updated but only sections with\n"
                "     <!-- AUTO-GENERATED-BEGIN:X --> ... <!-- AUTO-GENERATED-END:X -->\n"
                "     markers will be replaced.\n"
                "     To add markers, use --force-regenerate once."
            )

        new_content = update_existing_document(existing, dashboard, gt_missing, n_orphans)
        action = "update"

    if args.dry_run:
        print(f"\n{'─' * 60}")
        print("DRY RUN — would write:")
        print(f"{'─' * 60}")
        print(new_content[:3000] + "\n...[truncated]" if len(new_content) > 3000 else new_content)
        return 0

    COVERAGE_DOC.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_DOC.write_text(new_content)
    print(f"\n  ✅ {action}d: {COVERAGE_DOC.relative_to(ROOT)}")
    print(f"     {len(new_content)} chars, {new_content.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
