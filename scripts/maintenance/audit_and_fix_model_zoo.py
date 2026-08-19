#!/usr/bin/env python3
"""Audit and fix model zoo integrity issues.

Addresses:
1. Registry entries pointing to missing checkpoints (stale records)
2. Manifest entries with incorrect n_qubits (n_qubits=10 for unified/multiN models)
3. Registry ↔ Manifest inconsistency (registry has entries not in manifest)
4. Dashboard zoo_pass_rate divergence (zoo claims vs actual NPZ data)
5. Exclusion policy drift (dashboard vs training_exclusions.json)
6. Orphan checkpoints (on disk but not in manifest)
7. Models with pass_rate=0 that were never evaluated post-training

Usage:
    .venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py          # audit only
    .venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix    # apply safe fixes
    .venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --prune-stale
    .venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --archive-orphans
    .venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --coherence   # include zoo↔dashboard checks
    .venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ZOO_DIR = ROOT / "data" / "model_zoo"
MANIFEST_PATH = ZOO_DIR / "manifest.json"
REGISTRY_PATH = ZOO_DIR / "model_registry.json"
CHECKPOINTS_DIR = ZOO_DIR / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers — delegate to library for manifest, keep local for registry
# (model_registry.json has a different schema than ModelRegistryDB)
# ─────────────────────────────────────────────────────────────────────────────


def _load_manifest_entries() -> list[dict]:
    """Load raw manifest entries (dict form, not ZooEntry)."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("entries", data.get("models", []))


def _save_manifest_entries(entries: list[dict]) -> None:
    """Save raw manifest entries."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def _load_registry() -> dict:
    """Load model registry (model_id → record dict)."""
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    """Save model registry preserving key structure."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Audit functions
# ─────────────────────────────────────────────────────────────────────────────


def audit_manifest(entries: list[dict]) -> list[dict]:
    """Audit manifest entries for issues.

    Checks:
    - Missing checkpoint files on disk
    - n_qubits=10 for models named unified/multiN (registration bug)
    - Suspicious perfect pass_rate with few training points
    - SHA256 integrity (when hash stored)

    Returns list of findings: {entry_idx, checkpoint_file, issue, severity, fix_description}
    """
    findings = []

    for i, entry in enumerate(entries):
        cp = entry.get("checkpoint_file", "")
        nq = entry.get("n_qubits")
        notes = entry.get("notes", "")
        cp_path = CHECKPOINTS_DIR / cp

        # Check 1: checkpoint file exists on disk
        if cp and not cp_path.exists():
            findings.append(
                {
                    "entry_idx": i,
                    "checkpoint_file": cp,
                    "issue": "missing_checkpoint",
                    "severity": "critical",
                    "fix_description": f"Checkpoint {cp} not on disk (remove or regenerate)",
                }
            )
            continue  # Skip further checks for missing files

        # Check 2: n_qubits should be 0 for multi-N models
        if nq == 10 and ("unified" in cp.lower() or "multi" in cp.lower()):
            findings.append(
                {
                    "entry_idx": i,
                    "checkpoint_file": cp,
                    "issue": "n_qubits_10_for_unified_model",
                    "severity": "warning",
                    "fix_description": (
                        f"n_qubits=10 but uses unified architecture — "
                        f"likely save_mpnn_to_zoo bug. Notes: '{notes[:60]}'"
                    ),
                }
            )

        # Check 3: suspicious perfect pass_rate
        pr = entry.get("pass_rate", 0)
        pts = entry.get("n_training_points", 0)
        if pr == 1.0 and 0 < pts < 20:
            findings.append(
                {
                    "entry_idx": i,
                    "checkpoint_file": cp,
                    "issue": "suspicious_perfect_pass_rate",
                    "severity": "info",
                    "fix_description": (
                        f"pass_rate=1.0 with only {pts} points — "
                        f"may indicate overfitting or limited evaluation"
                    ),
                }
            )

        # Check 4: SHA256 integrity (if available and file exists)
        sha = entry.get("sha256", "")
        if sha and cp_path.exists():
            import hashlib

            actual = hashlib.sha256(cp_path.read_bytes()).hexdigest()
            if actual != sha:
                findings.append(
                    {
                        "entry_idx": i,
                        "checkpoint_file": cp,
                        "issue": "sha256_mismatch",
                        "severity": "critical",
                        "fix_description": (
                            f"SHA256 mismatch: manifest={sha[:12]}... disk={actual[:12]}... "
                            f"(corrupted or silently overwritten)"
                        ),
                    }
                )

    return findings


def audit_registry(registry: dict) -> list[dict]:
    """Audit model registry for issues.

    Checks:
    - Missing checkpoint files
    - multiN name but single-N training data
    - Empty training metrics
    """
    findings = []
    entries = list(registry.values()) if isinstance(registry, dict) else registry

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("model_id", "")
        cp = entry.get("checkpoint_path", "")

        # Check 1: missing checkpoint
        if cp:
            abs_cp = Path(cp) if Path(cp).is_absolute() else ROOT / cp
            if not abs_cp.exists():
                findings.append(
                    {
                        "model_id": mid,
                        "issue": "missing_checkpoint",
                        "severity": "warning",
                        "fix_description": f"Checkpoint not on disk: {cp}",
                    }
                )

        # Check 2: multiN name but single-N training data
        training = entry.get("training", {})
        n_vals = training.get("n_values_used", [])
        if ("multiN" in mid or "unified" in mid.lower()) and len(n_vals) == 1:
            findings.append(
                {
                    "model_id": mid,
                    "issue": "multiN_name_single_n_training",
                    "severity": "warning",
                    "fix_description": (
                        f"Name suggests multi-N but only trained on N={n_vals}. "
                        f"Likely registered via save_mpnn_to_zoo bug."
                    ),
                }
            )

        # Check 3: zero training metrics (never populated)
        tm = training.get("training_metrics", {})
        if tm and tm.get("final_mse") == 0 and tm.get("epochs") == 0:
            findings.append(
                {
                    "model_id": mid,
                    "issue": "empty_training_metrics",
                    "severity": "info",
                    "fix_description": "Training metrics all zeros (not populated at registration).",
                }
            )

    return findings


def audit_consistency(manifest_entries: list[dict], registry: dict) -> list[dict]:
    """Check manifest ↔ registry ↔ disk consistency."""
    findings = []

    manifest_files = {e.get("checkpoint_file", "") for e in manifest_entries}
    manifest_files.discard("")

    reg_entries = list(registry.values()) if isinstance(registry, dict) else registry
    reg_ids = set()
    for e in reg_entries:
        if isinstance(e, dict):
            reg_ids.add(e.get("model_id", ""))
    reg_ids.discard("")

    # Registry entries not in manifest
    in_reg_only = reg_ids - manifest_files
    if in_reg_only:
        findings.append(
            {
                "issue": "registry_entries_not_in_manifest",
                "severity": "info",
                "count": len(in_reg_only),
                "fix_description": (
                    f"{len(in_reg_only)} registry entries have no manifest counterpart "
                    f"(likely superseded). Use --prune-stale to clean."
                ),
                "details": sorted(in_reg_only)[:10],
            }
        )

    # Manifest entries not in registry
    in_manifest_only = manifest_files - reg_ids
    if in_manifest_only:
        findings.append(
            {
                "issue": "manifest_entries_not_in_registry",
                "severity": "warning",
                "count": len(in_manifest_only),
                "fix_description": (
                    f"{len(in_manifest_only)} manifest entries not in registry. "
                    f"Run `query_model_registry.py sync` to add."
                ),
                "details": sorted(in_manifest_only)[:10],
            }
        )

    # Orphan checkpoints on disk (excluding _archive/ and _best/)
    if CHECKPOINTS_DIR.exists():
        disk_files = {f.name for f in CHECKPOINTS_DIR.glob("*.pt")}
        orphans = disk_files - manifest_files
        if orphans:
            findings.append(
                {
                    "issue": "orphan_checkpoints_on_disk",
                    "severity": "warning",
                    "count": len(orphans),
                    "fix_description": (
                        f"{len(orphans)} .pt files not in manifest. "
                        f"Use --archive-orphans to move to _archive/."
                    ),
                    "details": sorted(orphans)[:10],
                }
            )

    return findings


def audit_coherence() -> list[dict]:
    """Check zoo ↔ dashboard ↔ exclusion policy coherence.

    This subsumes the checks from check_zoo_coherence.py so both can be
    run from a single entry point.
    """
    findings = []
    dashboard_path = ROOT / "data" / "model_quality_dashboard.json"

    if not dashboard_path.exists():
        findings.append(
            {
                "issue": "dashboard_missing",
                "severity": "info",
                "fix_description": "model_quality_dashboard.json not found — run update_cross_n_coverage.py",
            }
        )
        return findings

    with open(dashboard_path) as f:
        dashboard = json.load(f)
    configs = dashboard.get("configs", [])

    # Use library for manifest (ZooEntry objects)
    try:
        from qmbp_simulation.predictors.model_zoo import list_pretrained

        multi_n_entries = list_pretrained(n_qubits=0)
    except Exception as e:
        findings.append(
            {
                "issue": "library_import_failed",
                "severity": "critical",
                "fix_description": f"Cannot import model_zoo: {e}",
            }
        )
        return findings

    # ── Check: zoo pass_rate vs dashboard NPZ pass rates ─────────────────
    for entry in multi_n_entries:
        topo = entry.topology
        topo_configs = [c for c in configs if c.get("topology") == topo]
        if not topo_configs:
            continue

        best_npz_dual = max(
            (c.get("pass_rate_dual_criterion", 0) for c in topo_configs),
            default=0,
        )

        zoo_pr = entry.pass_rate
        if zoo_pr > 0 and best_npz_dual > 0:
            divergence = abs(zoo_pr - best_npz_dual)
            if divergence > 0.25:
                findings.append(
                    {
                        "issue": "zoo_dashboard_divergence",
                        "severity": "warning",
                        "checkpoint_file": entry.checkpoint_file,
                        "fix_description": (
                            f"{topo}: zoo_pass_rate={zoo_pr:.0%} vs "
                            f"best_npz_dual={best_npz_dual:.0%} (Δ={divergence:.0%}). "
                            f"Re-evaluate with evaluate_zoo_models.py"
                        ),
                    }
                )

        # Never-evaluated models (pass_rate=0 with training data)
        if zoo_pr == 0 and entry.n_training_points > 0:
            findings.append(
                {
                    "issue": "zoo_never_evaluated",
                    "severity": "warning",
                    "checkpoint_file": entry.checkpoint_file,
                    "fix_description": (
                        f"{topo}: {entry.n_training_points} pts but pass_rate=0 "
                        f"(never evaluated post-training). Run evaluate_zoo_models.py --update-zoo"
                    ),
                }
            )

        # Data freshness: more points available than model was trained on
        total_pts = sum(c.get("n_points", 0) for c in topo_configs)
        if entry.n_training_points > 0 and total_pts > entry.n_training_points * 1.5:
            findings.append(
                {
                    "issue": "zoo_data_expanded",
                    "severity": "info",
                    "checkpoint_file": entry.checkpoint_file,
                    "fix_description": (
                        f"{topo}: trained on {entry.n_training_points} pts "
                        f"but {total_pts} now available (+{total_pts - entry.n_training_points}). "
                        f"Retrain may improve generalization."
                    ),
                }
            )

        # Model trained on subset — more N values now available
        import re

        n_match = re.search(r"multiN_([\d+]+)_p", entry.checkpoint_file)
        if n_match:
            trained_n = set(int(x) for x in n_match.group(1).split("+"))
            available_n = set(
                c.get("n_qubits", 0) for c in topo_configs if c.get("training_utility") == "useful"
            )
            available_n.discard(0)
            new_n = available_n - trained_n
            if new_n and len(new_n) >= 2:
                findings.append(
                    {
                        "issue": "zoo_missing_n_coverage",
                        "severity": "info",
                        "checkpoint_file": entry.checkpoint_file,
                        "fix_description": (
                            f"{topo}: trained on N={sorted(trained_n)} but "
                            f"N={sorted(new_n)} also viable. Consider retraining."
                        ),
                    }
                )

    # ── Check: exclusion policy vs dashboard consistency ─────────────────
    try:
        from qmbp_simulation.analysis.metrics import get_excluded_files, load_training_exclusions

        excluded_files = get_excluded_files()
        registry = load_training_exclusions()
        excluded_entries = registry.get("excluded", [])

        # Dashboard says 'not_useful' but NOT in exclusion registry
        dashboard_not_useful = set()
        for c in configs:
            if c.get("training_utility") == "not_useful":
                fname = c.get("file", "")
                if fname:
                    dashboard_not_useful.add(fname)

        drift = dashboard_not_useful - excluded_files
        if drift:
            findings.append(
                {
                    "issue": "exclusion_policy_drift",
                    "severity": "info",
                    "count": len(drift),
                    "fix_description": (
                        f"{len(drift)} file(s) marked 'not_useful' in dashboard "
                        f"but NOT in training_exclusions.json. "
                        f"Run auto_detect_exclusions() to sync."
                    ),
                    "details": sorted(drift)[:5],
                }
            )

        # In exclusion registry but dashboard says it's useful
        useful_in_dash = set()
        for c in configs:
            utility = c.get("training_utility", "")
            if utility and utility != "not_useful":
                fname = c.get("file", "")
                if fname:
                    useful_in_dash.add(fname)

        stale_exclusions = excluded_files & useful_in_dash
        if stale_exclusions:
            findings.append(
                {
                    "issue": "stale_exclusion",
                    "severity": "warning",
                    "count": len(stale_exclusions),
                    "fix_description": (
                        f"{len(stale_exclusions)} file(s) excluded from training but dashboard "
                        f"now says they're useful. Data may have improved — remove from exclusions."
                    ),
                    "details": sorted(stale_exclusions)[:5],
                }
            )
    except ImportError:
        pass  # Exclusion check is optional

    # ── Check: retrain queue ─────────────────────────────────────────────
    try:
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        high_priority = [r for r in queue if r.get("priority", 99) <= 2]
        if high_priority:
            findings.append(
                {
                    "issue": "retrain_needed",
                    "severity": "info",
                    "count": len(high_priority),
                    "fix_description": (
                        f"{len(high_priority)} model(s) need retraining (priority 1-2): "
                        f"{', '.join(r.get('topology', '?') for r in high_priority[:3])}"
                    ),
                }
            )
    except Exception:
        pass  # Retrain check is best-effort

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Fix functions
# ─────────────────────────────────────────────────────────────────────────────


def fix_manifest_n_qubits(entries: list[dict], dry_run: bool = True) -> list[str]:
    """Tag n_qubits=10 unified models with 'single_n_eval' for clarity.

    These are legitimate single-N=10 evaluation exports, not true multi-N.
    The tag prevents dashboard from inflating zoo_pass_rate.
    """
    changes = []
    for i, entry in enumerate(entries):
        nq = entry.get("n_qubits")
        cp = entry.get("checkpoint_file", "")
        if nq == 10 and "unified" in cp.lower():
            tags = entry.get("tags", [])
            if "single_n_eval" not in tags:
                if not dry_run:
                    entry.setdefault("tags", []).append("single_n_eval")
                changes.append(f"Tagged [{i}] {cp} with 'single_n_eval'")
    return changes


def fix_registry_prune_stale(registry: dict, dry_run: bool = True) -> tuple[dict, list[str]]:
    """Remove registry entries whose checkpoints don't exist and are superseded.

    Only prunes entries where:
    - Checkpoint file doesn't exist on disk
    - A newer model for same topology exists (superseded) OR entry is a test

    Re-indexes with sequential string keys to match existing format ("0", "1", ...).
    """
    if not isinstance(registry, dict):
        return registry, []

    removed = []
    kept = []

    for entry in registry.values():
        if not isinstance(entry, dict):
            kept.append(entry)
            continue

        mid = entry.get("model_id", "")
        cp = entry.get("checkpoint_path", "")

        # Only prune if checkpoint is missing AND entry is superseded/test
        should_prune = False
        if cp:
            abs_cp = Path(cp) if Path(cp).is_absolute() else ROOT / cp
            if not abs_cp.exists():
                superseded_by = entry.get("superseded_by")
                status = entry.get("status", "")
                is_test = "test" in mid.lower()
                if superseded_by or status == "superseded" or is_test:
                    should_prune = True

        if should_prune:
            removed.append(mid)
        else:
            kept.append(entry)

    if dry_run:
        return registry, removed

    # Re-index with sequential string keys (matches existing format)
    new_reg = {str(i): e for i, e in enumerate(kept)}
    return new_reg, removed


def fix_archive_orphans(dry_run: bool = True) -> list[str]:
    """Move orphan checkpoint files to _archive/.

    Orphans are .pt files in checkpoints/ not tracked in the manifest.
    Moving to _archive/ keeps them recoverable without polluting the active set.
    """
    if not CHECKPOINTS_DIR.exists():
        return []

    manifest_entries = _load_manifest_entries()
    manifest_files = {e.get("checkpoint_file", "") for e in manifest_entries}
    disk_files = {f.name for f in CHECKPOINTS_DIR.glob("*.pt")}
    orphans = sorted(disk_files - manifest_files)

    if not orphans:
        return []

    archive_dir = CHECKPOINTS_DIR / "_archive"
    archived = []

    for fname in orphans:
        src = CHECKPOINTS_DIR / fname
        if not dry_run:
            archive_dir.mkdir(exist_ok=True)
            dst = archive_dir / fname
            shutil.move(str(src), str(dst))
        archived.append(fname)

    return archived


def fix_sync_exclusions(dry_run: bool = True) -> int:
    """Sync exclusion registry with dashboard 'not_useful' signals.

    Calls auto_detect_exclusions to pick up any new NPZ files that the
    dashboard flags but the exclusion registry hasn't captured yet.
    """
    try:
        from qmbp_simulation.analysis.metrics import auto_detect_exclusions

        new = auto_detect_exclusions(dry_run=dry_run)
        return len(new)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_report(
    manifest_findings: list[dict],
    registry_findings: list[dict],
    consistency_findings: list[dict],
    coherence_findings: list[dict] | None = None,
) -> None:
    """Print formatted audit report."""
    print("=" * 72)
    print("  MODEL ZOO INTEGRITY AUDIT")
    print(f"  {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 72)

    all_findings = manifest_findings + registry_findings + consistency_findings
    if coherence_findings:
        all_findings += coherence_findings
    total_issues = len(all_findings)

    # Manifest
    print(f"\n{'─' * 40}")
    print(f"  MANIFEST ({MANIFEST_PATH.relative_to(ROOT)})")
    print(f"{'─' * 40}")
    if manifest_findings:
        for f in manifest_findings:
            sev = f["severity"].upper()
            print(f"  [{sev}] {f.get('checkpoint_file', '')}: {f['issue']}")
            print(f"         {f['fix_description']}")
    else:
        print("  ✓ No issues found")

    # Registry
    print(f"\n{'─' * 40}")
    print(f"  REGISTRY ({REGISTRY_PATH.relative_to(ROOT)})")
    print(f"{'─' * 40}")
    if registry_findings:
        for f in registry_findings:
            sev = f["severity"].upper()
            print(f"  [{sev}] {f.get('model_id', '')}: {f['issue']}")
            print(f"         {f['fix_description']}")
    else:
        print("  ✓ No issues found")

    # Consistency
    print(f"\n{'─' * 40}")
    print("  CONSISTENCY")
    print(f"{'─' * 40}")
    if consistency_findings:
        for f in consistency_findings:
            sev = f["severity"].upper()
            count = f.get("count", "")
            print(f"  [{sev}] {f['issue']}" + (f" ({count})" if count else ""))
            print(f"         {f['fix_description']}")
    else:
        print("  ✓ All consistent")

    # Coherence (optional)
    if coherence_findings is not None:
        print(f"\n{'─' * 40}")
        print("  COHERENCE (zoo ↔ dashboard ↔ exclusions)")
        print(f"{'─' * 40}")
        if coherence_findings:
            for f in coherence_findings:
                sev = f["severity"].upper()
                print(f"  [{sev}] {f['issue']}")
                print(f"         {f['fix_description']}")
        else:
            print("  ✓ All coherent")

    # Summary
    print(f"\n{'─' * 40}")
    n_crit = sum(1 for f in all_findings if f.get("severity") == "critical")
    n_warn = sum(1 for f in all_findings if f.get("severity") == "warning")
    n_info = total_issues - n_crit - n_warn
    print(
        f"  SUMMARY: {total_issues} findings "
        f"(Critical: {n_crit}  Warning: {n_warn}  Info: {n_info})"
    )
    print(f"{'─' * 40}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fix", action="store_true", help="Apply fixes (default: audit only)")
    parser.add_argument(
        "--prune-stale", action="store_true", help="Remove stale registry entries (with --fix)"
    )
    parser.add_argument(
        "--archive-orphans",
        action="store_true",
        help="Move orphan checkpoints to _archive/ (with --fix)",
    )
    parser.add_argument(
        "--sync-exclusions",
        action="store_true",
        help="Sync exclusion registry with dashboard (with --fix)",
    )
    parser.add_argument(
        "--coherence", action="store_true", help="Include zoo↔dashboard↔exclusion coherence checks"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    args = parser.parse_args()

    # Load data
    manifest_entries = _load_manifest_entries()
    registry = _load_registry()

    # Run audits
    manifest_findings = audit_manifest(manifest_entries)
    registry_findings = audit_registry(registry)
    consistency_findings = audit_consistency(manifest_entries, registry)
    coherence_findings = audit_coherence() if args.coherence else None

    if args.json:
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "manifest_findings": manifest_findings,
            "registry_findings": registry_findings,
            "consistency_findings": consistency_findings,
        }
        if coherence_findings is not None:
            report["coherence_findings"] = coherence_findings
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(manifest_findings, registry_findings, consistency_findings, coherence_findings)

    # Apply fixes if requested
    if args.fix:
        print("\n" + "=" * 72)
        print("  APPLYING FIXES")
        print("=" * 72)

        # Fix 1: Tag manifest entries with n_qubits=10 unified models
        changes = fix_manifest_n_qubits(manifest_entries, dry_run=False)
        if changes:
            backup = MANIFEST_PATH.with_suffix(".json.bak")
            shutil.copy2(MANIFEST_PATH, backup)
            _save_manifest_entries(manifest_entries)
            print(f"\n  [MANIFEST] Tagged {len(changes)} entries:")
            for c in changes:
                print(f"    {c}")
            print(f"  Backup: {backup.relative_to(ROOT)}")
        else:
            print("\n  [MANIFEST] No changes needed")

        # Fix 2: Prune stale registry entries
        if args.prune_stale:
            backup = REGISTRY_PATH.with_suffix(".json.bak")
            shutil.copy2(REGISTRY_PATH, backup)
            new_reg, removed = fix_registry_prune_stale(registry, dry_run=False)
            if removed:
                _save_registry(new_reg)
                print(f"\n  [REGISTRY] Pruned {len(removed)} stale entries:")
                for r in removed[:10]:
                    print(f"    - {r}")
                if len(removed) > 10:
                    print(f"    ... and {len(removed) - 10} more")
                print(f"  Backup: {backup.relative_to(ROOT)}")
            else:
                print("\n  [REGISTRY] No stale entries to prune")
        else:
            _, would_remove = fix_registry_prune_stale(registry, dry_run=True)
            if would_remove:
                print(
                    f"\n  [REGISTRY] {len(would_remove)} entries would be pruned "
                    f"(use --prune-stale)"
                )

        # Fix 3: Archive orphan checkpoints
        if args.archive_orphans:
            archived = fix_archive_orphans(dry_run=False)
            if archived:
                print(f"\n  [ARCHIVE] Moved {len(archived)} orphans to _archive/:")
                for a in archived[:10]:
                    print(f"    → {a}")
                if len(archived) > 10:
                    print(f"    ... and {len(archived) - 10} more")
            else:
                print("\n  [ARCHIVE] No orphans to archive")
        else:
            would_archive = fix_archive_orphans(dry_run=True)
            if would_archive:
                print(f"\n  [ARCHIVE] {len(would_archive)} orphans found (use --archive-orphans)")

        # Fix 4: Sync exclusion registry
        if args.sync_exclusions:
            n_new = fix_sync_exclusions(dry_run=False)
            if n_new:
                print(f"\n  [EXCLUSIONS] Added {n_new} new exclusion(s)")
            else:
                print("\n  [EXCLUSIONS] Registry already in sync")
        else:
            n_would = fix_sync_exclusions(dry_run=True)
            if n_would:
                print(
                    f"\n  [EXCLUSIONS] {n_would} new exclusion(s) detected (use --sync-exclusions)"
                )

        print("\n  Done. Re-run without --fix to verify.")

    # Exit code: 1 if critical issues found
    all_findings = manifest_findings + registry_findings + consistency_findings
    if coherence_findings:
        all_findings += coherence_findings
    has_critical = any(f.get("severity") == "critical" for f in all_findings)
    return 1 if has_critical else 0


if __name__ == "__main__":
    sys.exit(main())
