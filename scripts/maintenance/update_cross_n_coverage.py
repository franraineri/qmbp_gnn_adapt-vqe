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
CROSS_TOPO_REPORT = ROOT / "internal" / "documentation" / "analysis" / "cross_topology_report.md"
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


# ═══════════════════════════════════════════════════════════════════════════════
# A. Quality Tier Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def compute_quality_tier_breakdown(npz_dir: Path) -> dict:
    """Compute quality tier distribution per NPZ file.
    
    Returns dict: {filename: {"verified": N, "approximate": M, "unverified": K, "total": N+M+K}}
    """
    if not npz_dir.exists():
        return {}
    
    breakdown = {}
    for npz_file in sorted(npz_dir.glob("*.npz")):
        try:
            data = np.load(str(npz_file), allow_pickle=True)
            tiers = data.get("quality_tier")
            if tiers is None:
                # Legacy NPZ without quality_tier field
                n_pts = len(data["h_values"])
                breakdown[npz_file.name] = {
                    "verified": 0,
                    "approximate": 0,
                    "unverified": n_pts,
                    "total": n_pts,
                    "legacy": True,
                }
            else:
                tier_list = list(tiers)
                breakdown[npz_file.name] = {
                    "verified": tier_list.count("verified"),
                    "approximate": tier_list.count("approximate"),
                    "unverified": tier_list.count("unverified"),
                    "total": len(tier_list),
                    "legacy": False,
                }
        except Exception as e:
            print(f"  ⚠️ Error reading {npz_file.name}: {e}", file=sys.stderr)
    return breakdown


def generate_quality_tier_table(tier_breakdown: dict) -> str:
    """Generate markdown table showing quality tier distribution."""
    if not tier_breakdown:
        return "*(No NPZ files found)*"
    
    rows = []
    total_verified = 0
    total_approx = 0
    total_unverified = 0
    n_legacy = 0
    
    for fname, counts in sorted(tier_breakdown.items()):
        total_verified += counts["verified"]
        total_approx += counts["approximate"]
        total_unverified += counts["unverified"]
        if counts.get("legacy"):
            n_legacy += 1
        
        # Only show files with >0 points
        if counts["total"] == 0:
            continue
            
        v_pct = counts["verified"] / counts["total"] * 100
        a_pct = counts["approximate"] / counts["total"] * 100
        u_pct = counts["unverified"] / counts["total"] * 100
        legacy_flag = " 📜" if counts.get("legacy") else ""
        
        rows.append(
            f"| {fname}{legacy_flag} | {counts['total']} | "
            f"{counts['verified']} ({v_pct:.0f}%) | "
            f"{counts['approximate']} ({a_pct:.0f}%) | "
            f"{counts['unverified']} ({u_pct:.0f}%) |"
        )
    
    total = total_verified + total_approx + total_unverified
    if total > 0:
        rows.append(
            f"| **TOTAL** | **{total}** | "
            f"**{total_verified}** ({total_verified/total*100:.0f}%) | "
            f"**{total_approx}** ({total_approx/total*100:.0f}%) | "
            f"**{total_unverified}** ({total_unverified/total*100:.0f}%) |"
        )
    
    lines = [
        "| File | Total | Verified ✅ | Approximate ⚠️ | Unverified ❓ |",
        "|------|-------|-------------|----------------|---------------|",
    ] + rows
    
    if n_legacy > 0:
        lines.append("")
        lines.append(f"*📜 = Legacy NPZ without quality_tier field ({n_legacy} files)*")
    
    return "\n".join(lines)


def check_quality_tier_warnings(tier_breakdown: dict) -> list[str]:
    """Check for quality tier issues and return warnings."""
    warnings = []
    
    for fname, counts in tier_breakdown.items():
        total = counts["total"]
        if total == 0:
            continue
        
        # Warning: >80% unverified
        unverified_ratio = counts["unverified"] / total
        if unverified_ratio > 0.80:
            warnings.append(
                f"⚠️ {fname}: {unverified_ratio:.0%} unverified — "
                f"run VQE refinement to generate quality data"
            )
        
        # Warning: legacy NPZ (no tier info)
        if counts.get("legacy") and total > 10:
            warnings.append(
                f"📜 {fname}: legacy NPZ ({total} pts) without quality_tier field. "
                f"Re-run with current pipeline to add tier metadata."
            )
        
        # Warning: mostly approximate, few verified
        if counts["approximate"] > 0 and counts["verified"] == 0 and total > 5:
            warnings.append(
                f"⚠️ {fname}: {counts['approximate']} approximate pts but 0 verified. "
                f"Consider running --refine-all to convert best predictions to verified."
            )
    
    return warnings


def compute_zoo_model_health() -> list[dict]:
    """Cross-integrate model_zoo manifest with NPZ quality tiers.

    Uses JSON manifest directly (no torch imports) to avoid segfaults.
    For each multi-N model in the zoo, computes:
    - Data quality score (from NPZ quality_tier field)
    - Retrain recommendation

    Returns list of dicts with model health info sorted by recommendation urgency.
    """
    manifest_path = DATA / "model_zoo" / "manifest.json"
    if not manifest_path.exists():
        return []

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception:
        return []

    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    # Focus on multi-N models (n_qubits == 0 → cross-N model)
    multi_n = [e for e in entries if e.get("n_qubits", -1) == 0]

    health_reports = []
    for entry in multi_n:
        topo = entry.get("topology", "")
        model_name = entry.get("model", "tfim_bond_resolved")
        p_layers = entry.get("p_layers", 1)
        pass_rate = entry.get("pass_rate", 0.0)
        n_training_points = entry.get("n_training_points", 0)

        # Compute quality from NPZ files directly (no torch needed)
        pattern = f"{topo}_N*_p{p_layers}.npz"
        npz_files = list(NPZ_DIR.glob(pattern)) if NPZ_DIR.exists() else []

        n_verified = 0
        n_approx = 0
        n_unverified = 0
        n_total = 0
        found = len(npz_files) > 0

        for npz_file in npz_files:
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                n_pts = len(data["h_values"])
                n_total += n_pts
                if "quality_tier" in data:
                    tiers = list(data["quality_tier"])
                    n_verified += tiers.count("verified")
                    n_approx += tiers.count("approximate")
                    n_unverified += tiers.count("unverified")
                else:
                    n_unverified += n_pts
            except Exception:
                continue

        # Compute quality score
        quality_score = 0.0
        if n_total > 0:
            quality_score = (n_verified * 1.0 + n_approx * 0.7 + n_unverified * 0.5) / n_total

        # Determine recommendation
        recommendation = "OK"
        urgency = 0

        if not found:
            recommendation = "ORPHAN: zoo model has no training data NPZ"
            urgency = 3
        elif quality_score < 0.55:
            recommendation = "RETRAIN: mostly unverified data (score < 0.55)"
            urgency = 3
        elif quality_score < 0.70:
            recommendation = "IMPROVE: run --refine-all to convert approx → verified"
            urgency = 2
        elif pass_rate < 0.5 and n_total > 20:
            recommendation = "INVESTIGATE: good data but low pass_rate"
            urgency = 2
        elif n_total < 15:
            recommendation = "EXPAND: too few training points for reliable cross-N"
            urgency = 1

        health_reports.append({
            "checkpoint": entry.get("checkpoint_file", "unknown"),
            "topology": topo,
            "n_training_points": n_training_points,
            "pass_rate": pass_rate,
            "quality_score": quality_score,
            "n_verified": n_verified,
            "n_approximate": n_approx,
            "n_unverified": n_unverified,
            "recommendation": recommendation,
            "urgency": urgency,
            "runner_tag": entry.get("runner_tag", ""),
            "date_tag": entry.get("date_tag", ""),
        })

    health_reports.sort(key=lambda x: (-x["urgency"], x["topology"]))
    return health_reports


def generate_zoo_health_table(health_reports: list[dict]) -> str:
    """Generate markdown table for zoo model health."""
    if not health_reports:
        return "*(No multi-N models in zoo)*"

    lines = [
        "| Model | Topology | Pts | Pass% | Q.Score | Verified | Recommendation |",
        "|-------|----------|-----|-------|---------|----------|----------------|",
    ]
    for r in health_reports:
        urgency_icon = {0: "✅", 1: "ℹ️", 2: "⚠️", 3: "🔴"}[r["urgency"]]
        lines.append(
            f"| `{r['checkpoint'][:45]}` | {r['topology']} | "
            f"{r['n_training_points']} | {r['pass_rate']:.0%} | "
            f"{r['quality_score']:.2f} | {r['n_verified']} | "
            f"{urgency_icon} {r['recommendation']} |"
        )
    return "\n".join(lines)


def _generate_zoo_health_section() -> str:
    """Wrapper that catches exceptions from compute_zoo_model_health."""
    try:
        reports = compute_zoo_model_health()
        return generate_zoo_health_table(reports)
    except Exception as e:
        return f"*(Error computing zoo health: {e})*"


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
    from qmbp_simulation.analysis.metrics import GAP_MASKING_THRESHOLD
    masked = [
        c for c in configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
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


def generate_training_health_table(dashboard: dict, gt_missing: dict, n_orphans: int, tier_breakdown: dict | None = None) -> str:
    """Training data health summary with quality tier breakdown."""
    from qmbp_simulation.analysis.metrics import classify_training_utility

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

    # Quality tier breakdown using classify_training_utility
    n_useful = 0
    n_insufficient = 0
    n_not_useful = 0
    quality_warnings = []
    for c in configs:
        category, reason = classify_training_utility(
            n_points=c.get("n_points", 0),
            pass_rate_dual=c.get("pass_rate_dual_criterion", 0.0),
            pass_rate_5pct=c.get("pass_rate_5pct", 0.0),
        )
        if category == "useful":
            n_useful += 1
        elif category == "insufficient_signal":
            n_insufficient += 1
            if c.get("n_points", 0) > 5:  # Only warn for non-trivial configs
                quality_warnings.append(
                    f"  ⚠️ {c['topology']}/N={c['n_qubits']}: {reason[:60]}..."
                )
        else:
            n_not_useful += 1
            quality_warnings.append(
                f"  ❌ {c['topology']}/N={c['n_qubits']}: {reason[:60]}..."
            )

    # Print quality warnings to stderr for visibility
    if quality_warnings:
        for w in quality_warnings[:10]:  # Limit to 10 warnings
            print(w, file=sys.stderr)
        if len(quality_warnings) > 10:
            print(f"  ... and {len(quality_warnings) - 10} more warnings", file=sys.stderr)

    # NPZ-level quality_tier field stats (if available)
    tier_stats = ""
    if tier_breakdown:
        total_verified = sum(t.get("verified", 0) for t in tier_breakdown.values())
        total_approx = sum(t.get("approximate", 0) for t in tier_breakdown.values())
        total_unverified = sum(t.get("unverified", 0) for t in tier_breakdown.values())
        total_tier_pts = total_verified + total_approx + total_unverified
        if total_tier_pts > 0:
            v_pct = total_verified / total_tier_pts * 100
            a_pct = total_approx / total_tier_pts * 100
            u_pct = total_unverified / total_tier_pts * 100
            tier_stats = f"✅{total_verified} ({v_pct:.0f}%) / ⚠️{total_approx} ({a_pct:.0f}%) / ❓{total_unverified} ({u_pct:.0f}%)"

    rows = [
        f"| Total NPZ files | {len(configs)} | |",
        f"| Total training points | {total_pts} | |",
        f"| **Quality: Useful** | {n_useful} configs | {'✅' if n_useful > 0 else '⚠️'} |",
        f"| **Quality: Insufficient** | {n_insufficient} configs | {'⚠️' if n_insufficient > 0 else '✅'} |",
        f"| **Quality: Not Useful** | {n_not_useful} configs | {'❌' if n_not_useful > 0 else '✅'} |",
    ]
    
    # Add NPZ quality_tier breakdown if available
    if tier_stats:
        tier_status = '✅' if total_verified / max(total_tier_pts, 1) > 0.5 else '⚠️'
        rows.append(f"| **NPZ Quality Tiers** | {tier_stats} | {tier_status} |")
    
    rows.extend([
        f"| NaN in θ | {n_nan} configs | {'✅' if n_nan == 0 else '❌'} |",
        f"| Zoo integrity | {zoo_ok} | {'✅' if zoo_ok else '❌'} |",
        f"| Zoo missing | {zoo_miss} | {'✅' if zoo_miss == 0 else '⚠️'} |",
        f"| Zoo orphan checkpoints | {n_orphans} | {'⚠️ cleanup needed' if n_orphans > 0 else '✅'} |",
        f"| GT coverage gaps | {total_missing_gt} uncovered h-points | {'⚠️' if total_missing_gt > 0 else '✅'} |",
        f"| Stale zoo models | {n_stale} | {'⚠️' if n_stale > 0 else '✅'} |",
        f"| Need retrain | {n_retrain} | {'🔄' if n_retrain > 0 else '✅'} |",
        f"| High θ discontinuity (>0.5) | {n_high_smooth} configs | {'⚠️' if n_high_smooth > 0 else '✅'} |",
        f"| Gap masking detected | {n_masked} configs | {'⚠️' if n_masked > 0 else '✅'} |",
    ])

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


def generate_document(dashboard: dict, gt_missing: dict, n_orphans: int, tier_breakdown: dict | None = None) -> str:
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
        generate_training_health_table(dashboard, gt_missing, n_orphans, tier_breakdown),
        f"<!-- AUTO-GENERATED-END:health -->",
        "",
    ]
    
    # Add quality tier breakdown table if available
    if tier_breakdown:
        lines.extend([
            "### Quality Tier Distribution (NPZ-level)",
            "",
            f"<!-- AUTO-GENERATED-BEGIN:quality_tiers -->",
            generate_quality_tier_table(tier_breakdown),
            f"<!-- AUTO-GENERATED-END:quality_tiers -->",
            "",
        ])
        # Add warnings
        tier_warnings = check_quality_tier_warnings(tier_breakdown)
        if tier_warnings:
            lines.extend([
                "**Quality Tier Warnings:**",
                "",
            ] + tier_warnings[:5] + (["", f"*(and {len(tier_warnings)-5} more)*"] if len(tier_warnings) > 5 else []) + [""])
    
    lines.extend([
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
    ])

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
        f"<!-- AUTO-GENERATED-BEGIN:large_n_extrapolation -->",
        generate_large_n_extrapolation_section(),
        f"<!-- AUTO-GENERATED-END:large_n_extrapolation -->",
        "",
        "---",
        "",
        "## Quality Tier Distribution",
        "",
        "Data quality breakdown by tier (verified=VQE-converged, approximate=MPNN-predicted, unverified=legacy):",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:tier_breakdown -->",
        generate_tier_breakdown(dashboard),
        f"<!-- AUTO-GENERATED-END:tier_breakdown -->",
        "",
        "---",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:training_plan -->",
        generate_training_plan(dashboard),
        f"<!-- AUTO-GENERATED-END:training_plan -->",
        "",
        "---",
        "",
        "## Model Zoo Health",
        "",
        "Cross-integration: model_zoo entries + NPZ quality tier scores.",
        "",
        f"<!-- AUTO-GENERATED-BEGIN:zoo_health -->",
        _generate_zoo_health_section(),
        f"<!-- AUTO-GENERATED-END:zoo_health -->",
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


# ═══════════════════════════════════════════════════════════════════════════════
# Large-N Extrapolation Section
# ═══════════════════════════════════════════════════════════════════════════════

EXTRAPOLATION_DIR = DATA / "large_n_extrapolation"


def generate_large_n_extrapolation_section() -> str:
    """Generate the large-N extrapolation summary from NPZ data and result JSONs.

    Scans data/large_n_extrapolation/*.npz for per-topology results and
    results/experiments/exp_large_n_extrap/run_*.json for speedup data.
    """
    if not EXTRAPOLATION_DIR.exists():
        return "No large-N extrapolation data found."

    npz_files = sorted(EXTRAPOLATION_DIR.glob("*.npz"))
    if not npz_files:
        return "No large-N extrapolation NPZ files found."

    # Parse NPZ files: {topology: [{N, h_range, n_pts, mean_de_gap, ...}]}
    topo_data: dict[str, list[dict]] = defaultdict(list)
    for npz_path in npz_files:
        # Parse filename: chain_1d_N30_p1.npz
        stem = npz_path.stem
        parts = stem.rsplit("_", 2)  # e.g. ["chain_1d", "N30", "p1"]
        if len(parts) < 3:
            continue
        topo = parts[0]
        n_str = parts[1]
        if not n_str.startswith("N"):
            continue
        n_qubits = int(n_str[1:])

        try:
            data = np.load(npz_path, allow_pickle=True)
            h_values = data["h_values"]
            n_pts = len(h_values)
            if n_pts == 0:
                continue

            e_key = "e_pred" if "e_pred" in data else ("e_vqe" if "e_vqe" in data else None)
            if e_key is None:
                continue

            e_pred = data[e_key].astype(float)
            e_exact = data["e_exact"].astype(float)
            gaps = data["gaps"].astype(float) if "gaps" in data else None

            abs_errs = np.abs(e_pred - e_exact)
            mean_abs_err = float(abs_errs.mean())
            per_site_err = mean_abs_err / max(n_qubits, 1)

            if gaps is not None:
                from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR
                de_gaps = abs_errs / np.maximum(gaps, 1e-10)
                mean_de_gap = float(de_gaps.mean())
                pass_5pct = int((de_gaps < DE_GAP_THRESHOLD).sum())
                dual_mask = (de_gaps < DE_GAP_THRESHOLD) & (abs_errs < MAX_ABS_ERROR)
                pass_dual = int(dual_mask.sum())
            else:
                mean_de_gap = -1
                pass_5pct = 0
                pass_dual = 0

            topo_data[topo].append({
                "n_qubits": n_qubits,
                "n_pts": n_pts,
                "h_min": float(h_values.min()),
                "h_max": float(h_values.max()),
                "mean_de_gap": mean_de_gap,
                "mean_abs_err": mean_abs_err,
                "per_site_err": per_site_err,
                "pass_5pct": pass_5pct,
                "pass_dual": pass_dual,
            })
        except Exception:
            continue

    if not topo_data:
        return "No valid large-N extrapolation data found."

    # Scan result JSONs for speedup data
    results_dir = ROOT / "results" / "experiments" / "exp_large_n_extrap"
    speedup_data: dict[str, dict[int, float]] = defaultdict(dict)  # topo -> {N: speedup}
    if results_dir.exists():
        for rfile in sorted(results_dir.glob("run_*.json")):
            try:
                with open(rfile) as f:
                    rdata = json.load(f)
                topo = rdata.get("config", {}).get("topology", "")
                results = rdata.get("results", {})
                # Find summary section (last section with comparison data)
                for sec_key in ["section_4", "section_3", "section_5"]:
                    sec = results.get(sec_key, {})
                    comp = sec.get("data", {}).get("comparison", {})
                    for n_str, entry in comp.items():
                        spd = entry.get("speedup")
                        if spd and spd > 1:
                            n_val = int(n_str)
                            # Keep the best (highest) speedup per (topo, N)
                            if n_val not in speedup_data[topo] or spd > speedup_data[topo][n_val]:
                                speedup_data[topo][n_val] = spd
            except Exception:
                continue

    # Build output
    lines = [
        "## Large-N Extrapolation (Zero-Shot)",
        "",
        "MPNN predictions at N >> training data. Model trained on N≤20,",
        "evaluated at N=30-100 via MPS backend. Speedup = VQE_evals / MPNN_evals.",
        "",
    ]

    # Per-topology table
    for topo in sorted(topo_data.keys()):
        entries = sorted(topo_data[topo], key=lambda x: x["n_qubits"])
        lines.append(f"### {topo}")
        lines.append("")
        lines.append("| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |")
        lines.append("|---|---------|-----|--------|--------|---------|-----------|---------|")

        for e in entries:
            n = e["n_qubits"]
            h_range = f"[{e['h_min']:.1f}, {e['h_max']:.1f}]"
            de_gap_str = f"{e['mean_de_gap']:.4f}" if e["mean_de_gap"] >= 0 else "—"
            per_site_str = f"{e['per_site_err']:.2e}"
            pass5_str = f"{e['pass_5pct']}/{e['n_pts']}"
            passd_str = f"{e['pass_dual']}/{e['n_pts']}"
            spd = speedup_data.get(topo, {}).get(n)
            spd_str = f"{spd:.0f}×" if spd else "—"

            lines.append(
                f"| {n} | {h_range} | {e['n_pts']} | {de_gap_str} | "
                f"{per_site_str} | {pass5_str} | {passd_str} | {spd_str} |"
            )
        lines.append("")

    # Extensive scaling summary
    lines.append("### Extensive Scaling Summary")
    lines.append("")
    lines.append("| Topology | N range | |ΔE|/N (mean) | Variation | Scaling |")
    lines.append("|----------|---------|--------------|-----------|---------|")

    for topo in sorted(topo_data.keys()):
        entries = sorted(topo_data[topo], key=lambda x: x["n_qubits"])
        if len(entries) < 2:
            continue
        per_site_errs = [e["per_site_err"] for e in entries]
        n_range = f"{entries[0]['n_qubits']}–{entries[-1]['n_qubits']}"
        mean_ps = np.mean(per_site_errs)
        variation = max(per_site_errs) / max(min(per_site_errs), 1e-10)
        scaling_ok = "✅ extensive" if variation < 3.0 else "⚠️ degrading"
        lines.append(
            f"| {topo} | {n_range} | {mean_ps:.2e} | {variation:.1f}× | {scaling_ok} |"
        )

    lines.append("")

    # ── MPNN vs Random VQE comparison table ──────────────────────────────
    # Extract per-topology, per-N comparison from result JSONs
    comparison_rows: list[dict] = []
    if results_dir.exists():
        for rfile in sorted(results_dir.glob("run_*.json")):
            try:
                with open(rfile) as f:
                    rdata = json.load(f)
                cfg = rdata.get("config", {})
                if cfg.get("skip_random_baseline", True):
                    continue  # No VQE baseline in this run
                topo = cfg.get("topology", "")
                results = rdata.get("results", {})
                # Get MPNN results
                mpnn_sec = results.get("section_2", {})
                mpnn_data = mpnn_sec.get("data", {}).get("mpnn_results", {})
                # Get VQE results
                vqe_data = {}
                for sk in ["section_3", "section_4"]:
                    sec = results.get(sk, {})
                    rd = sec.get("data", {}).get("random_results", {})
                    if rd:
                        vqe_data = rd
                        break
                if not vqe_data:
                    continue
                for n_str in mpnn_data:
                    if n_str not in vqe_data:
                        continue
                    m = mpnn_data[n_str]
                    v = vqe_data[n_str]
                    n_val = m.get("n_qubits", int(n_str))
                    # Keep the best (most recent) comparison per (topo, N)
                    comparison_rows.append({
                        "topo": topo,
                        "n": n_val,
                        "mpnn_de_gap": m.get("mean_de_gap", -1),
                        "mpnn_per_site": m.get("mean_abs_error_per_site", -1),
                        "mpnn_pass_dual": m.get("pass_rate_dual", 0),
                        "vqe_de_gap": v.get("mean_de_gap", -1),
                        "vqe_pass_dual": v.get("pass_rate_dual", 0),
                        "vqe_evals": v.get("total_evals", 0),
                        "mpnn_evals": m.get("n_points", 1),
                    })
            except Exception:
                continue

    if comparison_rows:
        # Deduplicate: keep best per (topo, N) by lowest mpnn_de_gap
        best_by_key: dict[tuple, dict] = {}
        for row in comparison_rows:
            key = (row["topo"], row["n"])
            if key not in best_by_key or row["mpnn_de_gap"] < best_by_key[key]["mpnn_de_gap"]:
                best_by_key[key] = row

        lines.append("### MPNN vs Random VQE vs Ground Truth")
        lines.append("")
        lines.append("Comparison at same h-points. MPNN: 1 forward pass (0 QPU). VQE: L-BFGS-B with random init.")
        lines.append("")
        lines.append("| Topology | N | MPNN ΔE/gap | VQE ΔE/gap | MPNN |ΔE|/N | MPNN wins? | Speedup | VQE evals |")
        lines.append("|----------|---|-------------|------------|------|-------|---------|-----------|")

        for key in sorted(best_by_key.keys()):
            row = best_by_key[key]
            mpnn_win = "✅" if row["mpnn_de_gap"] < row["vqe_de_gap"] else "❌"
            spd = row["vqe_evals"] / max(row["mpnn_evals"], 1)
            lines.append(
                f"| {row['topo']} | {row['n']} | "
                f"{row['mpnn_de_gap']:.4f} | {row['vqe_de_gap']:.4f} | "
                f"{row['mpnn_per_site']:.2e} | {mpnn_win} | "
                f"{spd:.0f}× | {row['vqe_evals']:,} |"
            )
        lines.append("")

        # Win rate summary
        n_wins = sum(1 for r in best_by_key.values() if r["mpnn_de_gap"] < r["vqe_de_gap"])
        total = len(best_by_key)
        lines.append(f"**MPNN win rate**: {n_wins}/{total} ({100*n_wins//max(total,1)}%)")
        lines.append("")

    # Key findings
    all_speedups = []
    for topo_speeds in speedup_data.values():
        all_speedups.extend(topo_speeds.values())
    if all_speedups:
        lines.append(f"**Speedup range**: {min(all_speedups):.0f}× – {max(all_speedups):.0f}×")
        lines.append("")

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


def generate_tier_breakdown(dashboard: dict) -> str:
    """Generate quality tier distribution per topology from NPZ data."""
    from collections import defaultdict as _ddict

    by_topo = _ddict(lambda: {"verified": 0, "approximate": 0, "unverified": 0, "total": 0})

    npz_dir = Path(ROOT) / "data" / "multi_n_training"
    if npz_dir.exists():
        for npz_file in sorted(npz_dir.glob("*.npz")):
            data = np.load(str(npz_file), allow_pickle=True)
            parts = npz_file.stem.split("_")
            n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
            if n_idx is None:
                continue
            topo = "_".join(parts[:n_idx])
            n_pts = len(data["h_values"])
            by_topo[topo]["total"] += n_pts

            if "quality_tier" in data:
                for t in data["quality_tier"].tolist():
                    key = str(t) if str(t) in ("verified", "approximate") else "unverified"
                    by_topo[topo][key] += 1
            else:
                by_topo[topo]["unverified"] += n_pts

    if not by_topo:
        return "*(No NPZ data found)*"

    rows = []
    for topo in sorted(by_topo.keys()):
        tb = by_topo[topo]
        total = tb["total"]
        if total == 0:
            continue
        rows.append(
            f"| {topo} | {total} | {tb['verified']} ({tb['verified']*100//total}%) | "
            f"{tb['approximate']} ({tb['approximate']*100//total}%) | "
            f"{tb['unverified']} ({tb['unverified']*100//total}%) |"
        )

    lines = [
        "| Topology | Total pts | Verified | Approximate | Unverified |",
        "|----------|-----------|----------|-------------|------------|",
    ] + rows
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

    updated, found = update_section(existing, "large_n_extrapolation", generate_large_n_extrapolation_section())
    if found:
        existing = updated

    updated, found = update_section(existing, "tier_breakdown", generate_tier_breakdown(dashboard))
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


def generate_cross_topology_report(
    dashboard: dict,
    tier_breakdown: dict | None = None,
) -> str:
    """Generate unified cross-topology report using exclusively pass_rate_dual.

    Data sources:
      - model_quality_dashboard.json (NPZ-level: pass_rate_dual, h_frontier)
      - data/model_zoo/manifest.json (zoo pass_rate = dual)
      - data/large_n_extrapolation/*.npz (extrapolation per-topology)
      - results/experiments/exp_large_n_extrap/run_*.json (speedup data)

    All quality metrics use dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).
    """
    configs = dashboard.get("configs", [])
    topo_sum = dashboard.get("topology_summary", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    topologies = sorted(topo_sum.keys())

    # ── Collect zoo data ──────────────────────────────────────────────────
    zoo_entries = _load_zoo_manifest()

    # ── Collect extrapolation data ────────────────────────────────────────
    extrap_data = _collect_extrapolation_data()

    # ── Build scorecard ───────────────────────────────────────────────────
    scorecard_rows = []
    for topo in topologies:
        topo_configs = sorted(
            [c for c in configs if c["topology"] == topo],
            key=lambda c: c["n_qubits"],
        )
        info = topo_sum[topo]

        # Training data quality
        total_pts = sum(c["n_points"] for c in topo_configs)
        verified_pts = 0
        if tier_breakdown:
            for fname, tb in tier_breakdown.items():
                if topo in fname:
                    verified_pts += tb.get("verified", 0)
        verified_pct = verified_pts * 100 // max(total_pts, 1)

        # N_max_viable with dual criterion (≥70% pass_dual)
        n_max_dual = _compute_n_max_viable_dual(topo_configs)

        # Best pass_rate_dual across all N
        best_dual = max((c.get("pass_rate_dual_criterion", 0) for c in topo_configs), default=0)

        # Zoo multi-N model
        zoo_multi = next(
            (e for e in zoo_entries if e.get("topology") == topo and e.get("n_qubits") == 0),
            None,
        )
        zoo_str = f"{zoo_multi['pass_rate']:.0%}" if zoo_multi else "—"
        zoo_icon = "✅" if zoo_multi and zoo_multi["pass_rate"] >= 0.70 else (
            "⚠️" if zoo_multi and zoo_multi["pass_rate"] >= 0.30 else "❌" if zoo_multi else "—"
        )

        # h_frontier (lowest h where pipeline still passes)
        h_frontiers = [c["h_frontier"] for c in topo_configs if c.get("h_frontier")]
        h_frontier_str = f"{min(h_frontiers):.2f}" if h_frontiers else "—"

        # Extrapolation best
        topo_extrap = extrap_data.get(topo, [])
        if topo_extrap:
            best_extrap_n = max(e["n_qubits"] for e in topo_extrap)
            best_extrap = next(e for e in topo_extrap if e["n_qubits"] == best_extrap_n)
            extrap_str = f"N={best_extrap_n}"
            extrap_dual = best_extrap["pass_dual"] / max(best_extrap["n_pts"], 1)
            if extrap_dual >= 0.70:
                extrap_str += " ✅"
            elif extrap_dual >= 0.30:
                extrap_str += f" ⚠️{extrap_dual:.0%}"
            else:
                extrap_str += f" ❌{extrap_dual:.0%}"
        else:
            extrap_str = "—"

        # Data quality icon
        dq_icon = "✅" if verified_pct >= 80 else ("⚠️" if verified_pct >= 40 else "❌")

        scorecard_rows.append({
            "topo": topo,
            "n_max_dual": n_max_dual,
            "best_dual": best_dual,
            "zoo_icon": zoo_icon,
            "zoo_str": zoo_str,
            "total_pts": total_pts,
            "verified_pct": verified_pct,
            "dq_icon": dq_icon,
            "extrap_str": extrap_str,
            "h_frontier": h_frontier_str,
        })

    # ── Build scaling matrix ──────────────────────────────────────────────
    all_n_values = sorted(set(c["n_qubits"] for c in configs))

    # ── Build document ────────────────────────────────────────────────────
    lines = [
        "# Cross-Topology Unified Report",
        "",
        f"**Generated**: {now}",
        "**Criterion**: `pass_rate_dual` (ΔE/gap < 5% AND |ΔE| < 0.10)",
        "**Model**: TFIM bond-resolved, HVA p=1",
        "",
        "> All quality metrics use the dual criterion exclusively.",
        "> `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality.",
        "",
        "---",
        "",
    ]

    # Section 1: Scorecard
    lines.extend(_build_scorecard_section(scorecard_rows))
    lines.append("")

    # Section 2: Scaling Matrix
    lines.extend(_build_scaling_matrix_section(topologies, configs, all_n_values))
    lines.append("")

    # Section 3: Gap Masking
    lines.extend(_build_gap_masking_section(configs))
    lines.append("")

    # Section 4: Extrapolation
    lines.extend(_build_extrapolation_section(extrap_data))
    lines.append("")

    # Section 5: Data Quality
    lines.extend(_build_data_quality_section(topologies, configs, tier_breakdown))
    lines.append("")

    # Section 6: Failure Mode Summary (lightweight — from dashboard only)
    lines.extend(_build_failure_summary_section(topologies, configs))
    lines.append("")

    # Section 7: Actions
    lines.extend(_build_actions_section(scorecard_rows, configs, topo_sum))
    lines.append("")

    return "\n".join(lines)


def _load_zoo_manifest() -> list[dict]:
    """Load zoo manifest entries."""
    manifest_path = DATA / "model_zoo" / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        data = json.load(open(manifest_path))
        return data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        return []


def _collect_extrapolation_data() -> dict[str, list[dict]]:
    """Collect extrapolation results from NPZ files. Returns {topo: [{n, pts, pass_dual, ...}]}."""
    if not EXTRAPOLATION_DIR.exists():
        return {}
    result: dict[str, list[dict]] = defaultdict(list)
    for npz_path in sorted(EXTRAPOLATION_DIR.glob("*.npz")):
        stem = npz_path.stem
        parts = stem.rsplit("_", 2)
        if len(parts) < 3 or not parts[1].startswith("N"):
            continue
        topo = parts[0]
        n_qubits = int(parts[1][1:])
        try:
            data = np.load(npz_path, allow_pickle=True)
            h_values = data["h_values"]
            n_pts = len(h_values)
            if n_pts == 0:
                continue
            e_key = "e_pred" if "e_pred" in data else ("e_vqe" if "e_vqe" in data else None)
            if e_key is None:
                continue
            e_pred = data[e_key].astype(float)
            e_exact = data["e_exact"].astype(float)
            gaps = data["gaps"].astype(float) if "gaps" in data else np.ones(n_pts)
            abs_errs = np.abs(e_pred - e_exact)
            de_gaps = abs_errs / np.maximum(gaps, 1e-10)
            dual_mask = (de_gaps < 0.05) & (abs_errs < 0.10)
            result[topo].append({
                "n_qubits": n_qubits,
                "n_pts": n_pts,
                "h_min": float(h_values.min()),
                "h_max": float(h_values.max()),
                "mean_de_gap": float(de_gaps.mean()),
                "mean_per_site": float(abs_errs.mean()) / max(n_qubits, 1),
                "pass_dual": int(dual_mask.sum()),
                "pass_5pct": int((de_gaps < 0.05).sum()),
            })
        except Exception:
            continue
    return dict(result)


def _compute_n_max_viable_dual(topo_configs: list[dict], threshold: float = 0.70) -> str:
    """Find largest N where pass_rate_dual >= threshold."""
    viable = [
        c["n_qubits"] for c in topo_configs
        if c.get("pass_rate_dual_criterion", 0) >= threshold
    ]
    return str(max(viable)) if viable else "—"


def _build_scorecard_section(rows: list[dict]) -> list[str]:
    lines = [
        "<!-- AUTO-GENERATED-BEGIN:scorecard -->",
        "## 1. Scorecard",
        "",
        "| Topology | N_max (dual≥70%) | Best pass_dual | Zoo model | Training pts | Data quality | Extrapolation | h_frontier |",
        "|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['topo']} | {r['n_max_dual']} | {r['best_dual']:.0%} | "
            f"{r['zoo_icon']} {r['zoo_str']} | {r['total_pts']} | "
            f"{r['dq_icon']} {r['verified_pct']}% | {r['extrap_str']} | {r['h_frontier']} |"
        )
    lines.append("<!-- AUTO-GENERATED-END:scorecard -->")
    return lines


def _build_scaling_matrix_section(
    topologies: list[str], configs: list[dict], all_n_values: list[int],
) -> list[str]:
    """Build pass_rate_dual matrix: topology × N."""
    # Filter to N values that at least 2 topologies have
    n_counts = defaultdict(int)
    for c in configs:
        n_counts[c["n_qubits"]] += 1
    useful_n = sorted(n for n, cnt in n_counts.items() if cnt >= 2)
    if not useful_n:
        useful_n = all_n_values[:8]

    lines = [
        "<!-- AUTO-GENERATED-BEGIN:scaling -->",
        "## 2. Scaling: pass_rate_dual per (Topology, N)",
        "",
        "| Topology | " + " | ".join(f"N={n}" for n in useful_n) + " |",
        "|----------|" + "|".join("---:" for _ in useful_n) + "|",
    ]
    for topo in topologies:
        row_cells = []
        for n in useful_n:
            match = next(
                (c for c in configs if c["topology"] == topo and c["n_qubits"] == n),
                None,
            )
            if match is None:
                row_cells.append("—")
            else:
                val = match.get("pass_rate_dual_criterion", 0)
                # Color-code via emoji
                if val >= 0.80:
                    row_cells.append(f"**{val:.0%}** ✅")
                elif val >= 0.50:
                    row_cells.append(f"{val:.0%} ⚠️")
                elif val > 0:
                    row_cells.append(f"{val:.0%} ❌")
                else:
                    row_cells.append("0%")
        lines.append(f"| {topo} | " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append("Legend: ✅ ≥80% | ⚠️ 50-79% | ❌ <50% | — no data")
    lines.append("<!-- AUTO-GENERATED-END:scaling -->")
    return lines


def _build_gap_masking_section(configs: list[dict]) -> list[str]:
    """Show gap masking severity per topology (reuses generate_gap_masking_table logic)."""
    masked = [
        c for c in configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > 0.10
    ]
    lines = [
        "<!-- AUTO-GENERATED-BEGIN:masking -->",
        "## 3. Gap Masking Severity",
        "",
        "Configs where single-criterion inflates by >10pp vs dual:",
        "",
    ]
    if not masked:
        lines.append("*No significant gap masking detected.*")
    else:
        lines.append("| Topology | N | pass@5% | pass@dual | Inflation |")
        lines.append("|----------|---|---------|-----------|-----------|")
        masked.sort(key=lambda c: -(c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0)))
        for c in masked[:15]:
            diff = c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0)
            lines.append(
                f"| {c['topology']} | {c['n_qubits']} | "
                f"{c['pass_rate_5pct']:.0%} | {c.get('pass_rate_dual_criterion', 0):.0%} | "
                f"+{diff:.0%} |"
            )
        if len(masked) > 15:
            lines.append(f"| ... | | | | *({len(masked) - 15} more)* |")

    lines.append("<!-- AUTO-GENERATED-END:masking -->")
    return lines


def _build_extrapolation_section(extrap_data: dict[str, list[dict]]) -> list[str]:
    """Extrapolation results using only dual criterion."""
    lines = [
        "<!-- AUTO-GENERATED-BEGIN:extrapolation -->",
        "## 4. Large-N Extrapolation (Zero-Shot)",
        "",
        "MPNN predictions at N >> training data. Only dual criterion reported.",
        "",
    ]
    if not extrap_data:
        lines.append("*No extrapolation data available yet.*")
        lines.append("<!-- AUTO-GENERATED-END:extrapolation -->")
        return lines

    lines.append("| Topology | N | h-range | Pts | pass_dual | |ΔE|/N | ΔE/gap (mean) |")
    lines.append("|----------|---|---------|-----|-----------|--------|---------------|")

    for topo in sorted(extrap_data.keys()):
        entries = sorted(extrap_data[topo], key=lambda x: x["n_qubits"])
        for e in entries:
            n = e["n_qubits"]
            dual_rate = e["pass_dual"] / max(e["n_pts"], 1)
            icon = "✅" if dual_rate >= 0.70 else ("⚠️" if dual_rate >= 0.30 else "❌")
            lines.append(
                f"| {topo} | {n} | [{e['h_min']:.1f}, {e['h_max']:.1f}] | "
                f"{e['n_pts']} | {e['pass_dual']}/{e['n_pts']} {icon} | "
                f"{e['mean_per_site']:.2e} | {e['mean_de_gap']:.4f} |"
            )

    lines.append("<!-- AUTO-GENERATED-END:extrapolation -->")
    return lines


def _build_data_quality_section(
    topologies: list[str], configs: list[dict], tier_breakdown: dict | None,
) -> list[str]:
    """Training data quality summary per topology."""
    lines = [
        "<!-- AUTO-GENERATED-BEGIN:data_quality -->",
        "## 5. Training Data Quality",
        "",
        "| Topology | NPZ files | Total pts | Verified | Approx | Unverified | Quality |",
        "|----------|-----------|-----------|----------|--------|------------|---------|",
    ]

    for topo in topologies:
        topo_configs = [c for c in configs if c["topology"] == topo]
        total_pts = sum(c["n_points"] for c in topo_configs)
        n_files = len(topo_configs)

        n_verified = 0
        n_approx = 0
        n_unverified = 0
        if tier_breakdown:
            for fname, tb in tier_breakdown.items():
                if topo in fname:
                    n_verified += tb.get("verified", 0)
                    n_approx += tb.get("approximate", 0)
                    n_unverified += tb.get("unverified", 0)

        pct = n_verified * 100 // max(total_pts, 1)
        icon = "✅" if pct >= 80 else ("⚠️" if pct >= 40 else "❌")

        lines.append(
            f"| {topo} | {n_files} | {total_pts} | "
            f"{n_verified} ({pct}%) | {n_approx} | {n_unverified} | {icon} |"
        )

    lines.append("<!-- AUTO-GENERATED-END:data_quality -->")
    return lines


def _build_failure_summary_section(topologies: list[str], configs: list[dict]) -> list[str]:
    """Lightweight failure mode classification from dashboard data only.

    Uses pass_rate_5pct - pass_rate_dual gap and training_utility to infer
    the dominant failure mode per topology. No NPZ raw reads needed.
    """
    from qmbp_simulation.analysis.metrics import GAP_MASKING_THRESHOLD

    lines = [
        "<!-- AUTO-GENERATED-BEGIN:failure_modes -->",
        "## 6. Failure Mode Classification",
        "",
        "| Topology | Mode | Evidence | Implication |",
        "|----------|------|----------|-------------|",
    ]

    for topo in topologies:
        topo_configs = [c for c in configs if c["topology"] == topo]
        best_dual = max((c.get("pass_rate_dual_criterion", 0) for c in topo_configs), default=0)

        if best_dual >= 0.95:
            lines.append(f"| {topo} | ✅ healthy | best_dual={best_dual:.0%} | Pipeline works correctly |")
            continue

        # Gap masking severity
        n_masked = sum(
            1 for c in topo_configs
            if c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
        )
        mask_severity = sum(
            c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0)
            for c in topo_configs
            if c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
        ) / max(n_masked, 1)

        # Training contamination indicators
        n_not_useful = sum(1 for c in topo_configs if c.get("training_utility") == "not_useful")
        n_insufficient = sum(1 for c in topo_configs if c.get("training_utility") == "insufficient_signal")

        # Classify
        if n_not_useful >= 3 and n_masked >= 3:
            mode = "🔴 contaminated"
            evidence = f"{n_not_useful} not-useful, {n_masked} gap-masked (sev={mask_severity:.0%})"
            implication = "Purge gap-masked data, retrain model"
        elif n_masked >= 2 and mask_severity > 0.30 and n_not_useful < 2:
            mode = "🔵 gap_masking"
            evidence = f"{n_masked} configs, severity={mask_severity:.0%}"
            implication = "Model works; |ΔE|>0.10 from N×ε (expected)"
        elif n_not_useful >= 2:
            mode = "⚠️ insufficient_data"
            evidence = f"{n_not_useful} not-useful, {n_insufficient} insufficient"
            implication = "More VQE refinement needed at viable h-range"
        elif best_dual < 0.50:
            # Check if verified per-site error is high (ansatz limit indicator)
            mean_abs = np.mean([c.get("mean_abs_error", 0) for c in topo_configs])
            avg_n = np.mean([c["n_qubits"] for c in topo_configs])
            per_site = mean_abs / max(avg_n, 1)
            if per_site > 0.015:
                mode = "⚠️ ansatz_limit"
                evidence = f"|ΔE|/N={per_site:.2e} (high), best_dual={best_dual:.0%}"
                implication = "HVA p=1 insufficient; needs p≥2 or fewer bonds"
            else:
                mode = "❓ unknown"
                evidence = f"best_dual={best_dual:.0%}, no clear pattern"
                implication = "Run --deep analyzer for detailed diagnosis"
        else:
            mode = "🟡 partial"
            evidence = f"best_dual={best_dual:.0%}, {n_masked} masked"
            implication = "Partially working; focus on viable h-range"

        lines.append(f"| {topo} | {mode} | {evidence} | {implication} |")

    lines.append("")
    lines.append("*Run `--deep` analyzer for full Tests A-L breakdown.*")
    lines.append("<!-- AUTO-GENERATED-END:failure_modes -->")
    return lines


def _build_actions_section(
    scorecard: list[dict], configs: list[dict], topo_sum: dict,
) -> list[str]:
    """Priority-ordered actions for each topology."""
    lines = [
        "<!-- AUTO-GENERATED-BEGIN:actions -->",
        "## 7. Recommended Actions",
        "",
    ]
    actions: list[tuple[int, str, str]] = []  # (priority, topo, action)

    for row in scorecard:
        topo = row["topo"]
        # Priority 1: Zoo model broken (pass_dual = 0 or < 30%)
        zoo_val = next(
            (e["pass_rate"] for e in _load_zoo_manifest()
             if e.get("topology") == topo and e.get("n_qubits") == 0),
            None,
        )
        if zoo_val is not None and zoo_val < 0.30:
            actions.append((1, topo, f"🔴 Re-train UnifiedMPNN (current pass_dual={zoo_val:.0%})"))

        # Priority 2: Low data quality
        if row["verified_pct"] < 40:
            actions.append((2, topo, f"⚠️ Run iterative-improve to increase verified% (currently {row['verified_pct']}%)"))

        # Priority 3: Missing extrapolation
        if row["extrap_str"] == "—":
            actions.append((3, topo, "ℹ️ Run large-N extrapolation to validate scaling"))

        # Priority 4: Good candidate for expansion
        if row["best_dual"] >= 0.80 and row["n_max_dual"] != "—":
            n_max = int(row["n_max_dual"])
            if n_max < 20:
                actions.append((4, topo, f"🟢 Expand to N={n_max + 4}: good candidate (pass_dual={row['best_dual']:.0%})"))

    actions.sort(key=lambda x: (x[0], x[1]))
    if actions:
        lines.append("| Priority | Topology | Action |")
        lines.append("|:---:|----------|--------|")
        for prio, topo, action in actions:
            lines.append(f"| {prio} | {topo} | {action} |")
    else:
        lines.append("*All topologies in good shape — no actions needed.*")

    lines.append("<!-- AUTO-GENERATED-END:actions -->")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Update accelerated_cross_n_coverage.md")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing")
    parser.add_argument(
        "--force-regenerate", action="store_true",
        help="Regenerate full document (discards manual sections). Default: update in-place.",
    )
    parser.add_argument(
        "--skip-tier-breakdown", action="store_true",
        help="Skip quality tier breakdown analysis (faster)",
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

    # Compute quality tier breakdown (per-NPZ)
    tier_breakdown = None
    if not args.skip_tier_breakdown:
        print("Computing quality tier breakdown...")
        tier_breakdown = compute_quality_tier_breakdown(NPZ_DIR)
        if tier_breakdown:
            n_legacy = sum(1 for t in tier_breakdown.values() if t.get("legacy"))
            n_verified = sum(t.get("verified", 0) for t in tier_breakdown.values())
            total_pts = sum(t.get("total", 0) for t in tier_breakdown.values())
            print(f"  {len(tier_breakdown)} NPZ files, {total_pts} total points")
            print(f"  ✅ {n_verified} verified ({n_verified*100//max(total_pts,1)}%)")
            if n_legacy:
                print(f"  📜 {n_legacy} legacy NPZ (no tier field)")
            # Check for quality warnings
            tier_warnings = check_quality_tier_warnings(tier_breakdown)
            if tier_warnings:
                print(f"  ⚠️  {len(tier_warnings)} quality warnings (see report)")

    # Determine update strategy
    if args.force_regenerate or not COVERAGE_DOC.exists():
        print("Generating full document...")
        new_content = generate_document(dashboard, gt_missing, n_orphans, tier_breakdown)
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

    # ── Generate unified cross-topology report ────────────────────────────
    print("\nGenerating cross-topology unified report...")
    cross_topo_content = generate_cross_topology_report(dashboard, tier_breakdown)
    if not args.dry_run:
        CROSS_TOPO_REPORT.parent.mkdir(parents=True, exist_ok=True)
        CROSS_TOPO_REPORT.write_text(cross_topo_content)
        print(f"  ✅ written: {CROSS_TOPO_REPORT.relative_to(ROOT)}")
        print(f"     {len(cross_topo_content)} chars, {cross_topo_content.count(chr(10))} lines")
    else:
        print("  (dry-run — not written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
