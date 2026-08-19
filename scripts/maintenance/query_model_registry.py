#!/usr/bin/env python3
"""Query the model registry DB from command line.

Subcommands:
    list       List models with filters
    get        Get detailed info for a specific model
    summary    Show registry statistics
    history    Show event history (audit trail)
    regressions  Detect and display regressions
    timeline   Show chronological timeline for a model
    sync       Sync registry from zoo manifest + NPZ + dashboard enrichment
    tag        Add or remove tags from a model
    tags       List all tags or query models by tag
    validate   Validate model integrity
    health     Show training data health for a model
    best       Find the best model for deployment
    versions   List models with version info
    version    Show version info for a specific model
    diagnose   Run failure diagnostics for a model
    diagnostics  Run failure diagnostics for all models
    comprehensive-health  Full health report combining integrity + quality + diagnostics
    health-dashboard  Generate full health dashboard for all models

Examples:
    .venv/bin/python scripts/maintenance/query_model_registry.py list
    .venv/bin/python scripts/maintenance/query_model_registry.py list --topology chain_1d
    .venv/bin/python scripts/maintenance/query_model_registry.py list --min-points 100 --json
    .venv/bin/python scripts/maintenance/query_model_registry.py get "unified_tfim_br_chain_1d*"
    .venv/bin/python scripts/maintenance/query_model_registry.py summary
    .venv/bin/python scripts/maintenance/query_model_registry.py history --model-id "unified_tfim_br_chain_1d*"
    .venv/bin/python scripts/maintenance/query_model_registry.py history --event-type regression_detected
    .venv/bin/python scripts/maintenance/query_model_registry.py regressions
    .venv/bin/python scripts/maintenance/query_model_registry.py timeline "unified_tfim_br_chain_1d_multiN_6+8+10+12+20_p1.pt"
    .venv/bin/python scripts/maintenance/query_model_registry.py sync

    # Tagging commands :
    .venv/bin/python scripts/maintenance/query_model_registry.py tag <model_id> --add production
    .venv/bin/python scripts/maintenance/query_model_registry.py tag <model_id> --remove experimental
    .venv/bin/python scripts/maintenance/query_model_registry.py tags
    .venv/bin/python scripts/maintenance/query_model_registry.py tags --query production

    # Validation commands (Improvements #4, #8, #9):
    .venv/bin/python scripts/maintenance/query_model_registry.py validate
    .venv/bin/python scripts/maintenance/query_model_registry.py validate <model_id>
    .venv/bin/python scripts/maintenance/query_model_registry.py health <model_id>
    .venv/bin/python scripts/maintenance/query_model_registry.py best -t chain_1d -n 20
    .venv/bin/python scripts/maintenance/query_model_registry.py best -t ladder -n 30 --require-tag production

    # Versioning commands :
    .venv/bin/python scripts/maintenance/query_model_registry.py versions
    .venv/bin/python scripts/maintenance/query_model_registry.py versions --topology chain_1d
    .venv/bin/python scripts/maintenance/query_model_registry.py version <model_id>
    .venv/bin/python scripts/maintenance/query_model_registry.py version <model_id> --chain

    # Failure diagnostics commands :
    .venv/bin/python scripts/maintenance/query_model_registry.py diagnose <model_id>
    .venv/bin/python scripts/maintenance/query_model_registry.py diagnose <model_id> --force
    .venv/bin/python scripts/maintenance/query_model_registry.py diagnostics
    .venv/bin/python scripts/maintenance/query_model_registry.py diagnostics --topology chain_1d
    .venv/bin/python scripts/maintenance/query_model_registry.py comprehensive-health <model_id>
    .venv/bin/python scripts/maintenance/query_model_registry.py health-dashboard
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB


def cmd_list(args, db: ModelRegistryDB):
    """List models matching filters."""
    if args.model_id:
        results = [
            r
            for r in db.list_all(include_archived=args.include_archived)
            if fnmatch.fnmatch(r.model_id, args.model_id)
        ]
    else:
        results = db.query(
            topology=args.topology,
            model_name=args.model_name,
            min_training_points=args.min_points,
            min_n_values=args.min_n_values,
            status=None if args.include_archived else "active",
        )

    if args.json:
        print(json.dumps([ModelRegistryDB._serialize(r) for r in results], indent=2))
        return

    if not results:
        print("  No models match the query.")
        return

    print(f"\n  Found {len(results)} model(s):\n")
    for r in results:
        _print_model_card(r)


def cmd_get(args, db: ModelRegistryDB):
    """Get detailed info for a specific model."""
    results = [
        r for r in db.list_all(include_archived=True) if fnmatch.fnmatch(r.model_id, args.model_id)
    ]

    if not results:
        print(f"  Model not found: {args.model_id}")
        return

    if args.json:
        print(json.dumps([ModelRegistryDB._serialize(r) for r in results], indent=2))
        return

    for r in results:
        _print_model_card(r, verbose=True)


def cmd_summary(args, db: ModelRegistryDB):
    """Show registry statistics."""
    s = db.summary()
    hs = db.history_summary()

    if args.json:
        print(json.dumps({"registry": s, "history": hs}, indent=2))
        return

    print(f"\n  ╔{'═' * 48}╗")
    print(f"  ║  Model Registry Summary{'':>23}║")
    print(f"  ╠{'═' * 48}╣")
    print(f"  ║  Total models:          {s['total_models']:<22}║")
    print(f"  ║  Active models:         {s['active_models']:<22}║")
    print(f"  ║  Topologies:            {', '.join(s['topologies']):<22}║")
    print(f"  ║  Model types:           {', '.join(s['model_names']):<22}║")
    print(f"  ║  Total training points: {s['total_training_points']:<22}║")
    print(f"  ║  Max N trained:         {s['max_n_trained']:<22}║")
    print(f"  ╠{'═' * 48}╣")
    print(f"  ║  History events:        {hs.get('total_events', 0):<22}║")
    print(f"  ║  Regressions detected:  {hs.get('regressions_detected', 0):<22}║")
    if hs.get("last_event"):
        print(f"  ║  Last event:            {hs['last_event'][:19]:<22}║")
    print(f"  ╚{'═' * 48}╝\n")

    # Per-topology breakdown table
    active = db.query()
    if active:
        topos = sorted(set(r.topology for r in active))
        print(f"  {'Topology':<14} {'Models':<8} {'Total pts':<12} {'N values':<20}")
        print(f"  {'─' * 14} {'─' * 8} {'─' * 12} {'─' * 20}")
        for topo in topos:
            topo_models = [r for r in active if r.topology == topo]
            total_pts = sum(r.training.total_training_points for r in topo_models)
            all_n = sorted(set(n for r in topo_models for n in r.training.n_values_used))
            n_str = "+".join(str(n) for n in all_n)
            print(f"  {topo:<14} {len(topo_models):<8} {total_pts:<12} {n_str:<20}")
        print()


def cmd_history(args, db: ModelRegistryDB):
    """Show event history."""
    model_id = None
    if args.model_id:
        # Support wildcards — find matching models then filter history
        all_models = db.list_all(include_archived=True)
        matching_ids = [
            r.model_id for r in all_models if fnmatch.fnmatch(r.model_id, args.model_id)
        ]
        if not matching_ids:
            print(f"  No models match: {args.model_id}")
            return
        # Get history for all matching
        events = []
        for mid in matching_ids:
            events.extend(db.get_history(model_id=mid, event_type=args.event_type, limit=None))
        events.sort(key=lambda e: e.timestamp, reverse=True)
        if args.limit:
            events = events[: args.limit]
    else:
        events = db.get_history(
            event_type=args.event_type,
            topology=args.topology,
            limit=args.limit or 50,
        )

    if args.json:
        from dataclasses import asdict

        print(json.dumps([asdict(e) for e in events], indent=2))
        return

    if not events:
        print("  No history events found.")
        return

    print(f"\n  History ({len(events)} events):\n")
    for e in events:
        icon = _event_icon(e.event_type)
        details_str = ""
        if e.details:
            key_items = []
            if "pass_rate_dual" in e.details:
                key_items.append(f"pass={e.details['pass_rate_dual']:.0%}")
            if "delta" in e.details:
                key_items.append(f"Δ={e.details['delta']:+.0%}")
            if "training_points" in e.details:
                key_items.append(f"pts={e.details['training_points']}")
            if "new_training_points" in e.details:
                key_items.append(f"pts={e.details['new_training_points']}")
            if "superseded_by" in e.details:
                key_items.append(f"by={e.details['superseded_by'][:30]}")
            if "notes" in e.details and e.details["notes"]:
                key_items.append(e.details["notes"][:40])
            if key_items:
                details_str = f"  ({', '.join(key_items)})"

        ts = e.timestamp[:19] if len(e.timestamp) > 19 else e.timestamp
        model_short = e.model_id[:45] if len(e.model_id) > 45 else e.model_id
        print(f"  {icon} {ts}  {e.event_type:<22} {model_short}{details_str}")
    print()


def cmd_regressions(args, db: ModelRegistryDB):
    """Detect and display regressions."""
    regressions = db.detect_regressions(threshold=args.threshold)

    if args.json:
        print(json.dumps(regressions, indent=2))
        return

    if not regressions:
        print("  ✅ No regressions detected.")
        return

    print(f"\n  ⚠️  {len(regressions)} regression(s) detected:\n")
    for reg in regressions:
        print(f"  ❌ {reg['model_id'][:50]}")
        print(f"     Topology: {reg['topology']}")
        print(
            f"     Pass rate: {reg['prev_pass_rate']:.0%} → {reg['curr_pass_rate']:.0%} (Δ={reg['delta']:+.0%})"
        )
        print(f"     At: {reg['evaluated_at'][:19]}")
        print()


def cmd_timeline(args, db: ModelRegistryDB):
    """Show chronological timeline for a model."""
    timeline = db.get_model_timeline(args.model_id)

    if not timeline:
        print(f"  No events found for: {args.model_id}")
        return

    if args.json:
        print(json.dumps(timeline, indent=2))
        return

    # Also show current state
    record = db.get_model(args.model_id)
    if record:
        print(f"\n  Model: {record.model_id}")
        print(
            f"  Status: {record.status}  |  Topology: {record.topology}  |  Points: {record.training.total_training_points}"
        )
        print(f"  N values: {record.training.n_values_used}")
        print()

    print(f"  Timeline ({len(timeline)} events):\n")
    for i, entry in enumerate(timeline):
        ts = entry.get("timestamp", "?")[:19]
        event = entry.pop("timestamp", "")
        event_type = entry.pop("event", "?")
        icon = _event_icon(event_type)
        connector = "├─" if i < len(timeline) - 1 else "└─"
        print(f"  {connector} {icon} {ts}  {event_type}")
        for k, v in entry.items():
            if v and k != "notes":
                print(f"  │     {k}: {v}")
        if entry.get("notes"):
            print(f"  │     notes: {entry['notes'][:60]}")
    print()


def cmd_sync(args, db: ModelRegistryDB):
    """Sync registry from zoo manifest + NPZ enrichment + dashboard + training curves."""
    print("  Syncing from zoo manifest...")
    result = db.sync_from_manifest()
    print(f"  → Added: {result['added']}, Skipped: {result['skipped']}")

    print("  Enriching with NPZ point counts...")
    enriched = db.enrich_points_per_n()
    print(f"  → Enriched: {enriched} records")

    print("  Enriching from dashboard...")
    dashboard_enriched = db.enrich_from_dashboard()
    print(f"  → Dashboard enriched: {dashboard_enriched} records")

    # ── Training curves cross-check ──────────────────────────────────────
    print("  Cross-checking training curves vs registry MSE...")
    from qmbp_simulation.analysis.metrics import validate_data_consistency
    dc = validate_data_consistency()
    reg_curves = dc.get("registry_vs_curves", {})
    if reg_curves:
        n_consistent = sum(1 for v in reg_curves.values() if v["consistent"])
        n_total = len(reg_curves)
        print(f"  → Curve matches: {n_consistent}/{n_total} consistent")
        for model_id, info in reg_curves.items():
            if not info["consistent"]:
                print(
                    f"    ⚠️ {model_id[:40]}: reg_mse={info['registry_mse']:.4e} "
                    f"vs curve={info['curve_final_mse']:.4e} "
                    f"(epochs: {info['n_epochs_registry']} vs {info['n_epochs_curve']})"
                )
    else:
        print("  → No training curve matches found")

    # ── Backfill pass_rate_by_n ──────────────────────────────────────────
    print("  Backfilling pass_rate_by_n from comparisons...")
    from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons
    n_backfilled = backfill_pass_rate_by_n_from_comparisons()
    print(f"  → Backfilled: {n_backfilled} models")

    s = db.summary()
    print(
        f"\n  Registry: {s['active_models']} active models, {s['total_training_points']} total points"
    )


# ─── New Commands (Improvements #3, #4, #8, #9) ─────────────────────────────


def cmd_tag(args, db: ModelRegistryDB):
    """Add or remove tags from models."""
    if args.add:
        success = db.add_tag(args.model_id, args.add)
        if success:
            print(f"  ✓ Added tag '{args.add}' to {args.model_id}")
        else:
            print("  ✗ Tag already exists or model not found")
    elif args.remove:
        success = db.remove_tag(args.model_id, args.remove)
        if success:
            print(f"  ✓ Removed tag '{args.remove}' from {args.model_id}")
        else:
            print("  ✗ Tag not found or model not found")
    elif args.list_tags:
        record = db.get_model(args.model_id)
        if record:
            if args.json:
                print(json.dumps(record.tags))
            else:
                print(f"  Tags for {args.model_id}: {', '.join(record.tags) or '(none)'}")
        else:
            print(f"  Model not found: {args.model_id}")


def cmd_tags(args, db: ModelRegistryDB):
    """List all tags or query models by tag."""
    if args.query:
        results = db.query_by_tag(args.query, include_archived=args.include_archived)
        if args.json:
            print(json.dumps([db._serialize(r) for r in results], indent=2))
        else:
            if not results:
                print(f"  No models with tag '{args.query}'")
            else:
                print(f"\n  Models with tag '{args.query}':\n")
                for r in results:
                    print(f"    • {r.model_id} ({r.topology})")
    else:
        all_tags = db.list_all_tags()
        if args.json:
            print(json.dumps(all_tags, indent=2))
        else:
            if not all_tags:
                print("  No tags defined")
            else:
                print("\n  All tags:\n")
                for tag, count in all_tags.items():
                    print(f"    {tag}: {count} model(s)")


def cmd_validate(args, db: ModelRegistryDB):
    """Validate model integrity."""
    if args.model_id:
        report = db.validate_integrity(args.model_id)
        if args.json:
            from dataclasses import asdict

            print(json.dumps(asdict(report), indent=2))
        else:
            icon = "✅" if report.all_ok else "❌"
            print(f"\n  {icon} Integrity check for {args.model_id}:\n")
            print(f"    Checkpoint exists: {'✓' if report.checkpoint_exists else '✗'}")
            print(f"    Hash matches:      {'✓' if report.hash_matches else '✗'}")
            print(f"    Manifest OK:       {'✓' if report.manifest_consistent else '✗'}")
            if report.issues:
                print("\n    Issues:")
                for issue in report.issues:
                    print(f"      • {issue}")
    else:
        # Validate all
        results = db.validate_all_integrity()
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n  Integrity validation complete:\n")
            print(f"    Checked:     {results['n_checked']}")
            print(f"    OK:          {results['n_ok']} ✅")
            print(f"    With issues: {results['n_issues']} ❌")
            if results["models_with_issues"]:
                print("\n    Models with issues:")
                for m in results["models_with_issues"]:
                    print(f"      • {m['model_id']}")
                    for issue in m["issues"]:
                        print(f"        - {issue}")


def cmd_health(args, db: ModelRegistryDB):
    """Show training data health for a model."""
    health = db.get_training_health(args.model_id)

    if args.json:
        print(json.dumps(health, indent=2))
        return

    if "error" in health:
        print(f"  ✗ {health['error']}")
        return

    icon = {"use": "✅", "investigate": "⚠️ ", "retrain": "❌"}.get(health["recommendation"], "?")

    print(f"\n  {icon} Health report for {health['model_id']}:\n")
    print(f"    Topology:          {health['topology']}")
    print(f"    Training utility:  {health['training_utility'] or 'unknown'}")
    print(f"    Latest pass_rate:  {health['latest_pass_rate']:.0%}")
    print(f"    Needs retrain:     {'Yes' if health['needs_retrain'] else 'No'}")
    print(f"    Model stale:       {'Yes' if health['model_stale'] else 'No'}")
    print(f"    Dashboard synced:  {health['dashboard_synced']}")
    print(f"    Recommendation:    {health['recommendation'].upper()}")

    if health["quality_issues"]:
        print("\n    Issues:")
        for issue in health["quality_issues"]:
            print(f"      • {issue}")


def cmd_best(args, db: ModelRegistryDB):
    """Find the best model for deployment."""
    record = db.get_best_for_deployment(
        topology=args.topology,
        model_name=args.model_name,
        p_layers=args.p_layers,
        n_target=args.n_target,
        require_tag=args.require_tag,
    )

    if args.json:
        if record:
            print(json.dumps(db._serialize(record), indent=2))
        else:
            print(json.dumps(None))
        return

    if record is None:
        print("\n  ✗ No suitable model found for:")
        print(f"    Topology: {args.topology}")
        print(f"    Model:    {args.model_name}")
        print(f"    p_layers: {args.p_layers}")
        print(f"    N target: {args.n_target}")
        return

    print(f"\n  ✅ Best model for N={args.n_target}:\n")
    _print_model_card(record, verbose=True)


# ─── Versioning Commands  ───────────────────────────────────


def cmd_versions(args, db: ModelRegistryDB):
    """List models with version information."""
    versions = db.list_versions(
        topology=args.topology,
        model_name=args.model_name,
        p_layers=args.p_layers,
    )

    if args.json:
        print(json.dumps(versions, indent=2))
        return

    if not versions:
        print("  No models found.")
        return

    print(f"\n  Model Versions ({len(versions)} models):\n")
    print(
        f"  {'Topology':<14} {'Model':<22} {'p':<3} {'Ver':<5} {'Hist':<5} {'Points':<8} {'Model ID':<40}"
    )
    print(f"  {'─' * 14} {'─' * 22} {'─' * 3} {'─' * 5} {'─' * 5} {'─' * 8} {'─' * 40}")

    for v in versions:
        model_id_short = v["model_id"][:38] + ".." if len(v["model_id"]) > 40 else v["model_id"]
        print(
            f"  {v['topology']:<14} {v['model_name']:<22} {v['p_layers']:<3} "
            f"v{v['version']:<4} {v['n_versions_in_history']:<5} "
            f"{v['training_points']:<8} {model_id_short:<40}"
        )
    print()


def cmd_version(args, db: ModelRegistryDB):
    """Show version info for a specific model."""
    if args.chain:
        # Show full version chain
        chain = db.get_version_chain(args.model_id)

        if args.json:
            print(json.dumps(chain, indent=2))
            return

        if not chain:
            print(f"  Model not found: {args.model_id}")
            return

        print(f"\n  Version Chain for {args.model_id}:\n")
        for entry in chain:
            icon = "→" if entry["version"] < len(chain) else "★"
            status_icon = {"active": "✓", "archived": "📦", "superseded": "⏭️ "}.get(
                entry["status"], "?"
            )
            ts = entry["created"][:19] if entry["created"] != "unknown" else "unknown"
            print(f"  {icon} v{entry['version']}  {status_icon} {entry['model_id'][:50]}")
            print(f"       Created: {ts}  |  Points: {entry['training_points']}")
        print()
    else:
        # Show version info for single model
        info = db.get_version_info(args.model_id)

        if args.json:
            print(json.dumps(info, indent=2))
            return

        if info is None:
            print(f"  Model not found: {args.model_id}")
            return

        latest_icon = "✅ (latest)" if info["is_latest"] else "⚠️  (superseded)"

        print(f"\n  Version Info for {info['model_id']}:\n")
        print(f"    Version:         v{info['version']} {latest_icon}")
        print(f"    Topology:        {info['topology']}")
        print(f"    Created:         {info['created'][:19] if info['created'] else 'unknown'}")
        print(f"    History length:  {len(info['version_history'])} previous version(s)")

        if info["version_history"]:
            print("    Previous IDs:")
            for i, hist_id in enumerate(info["version_history"]):
                print(f"      v{i + 1}: {hist_id[:60]}")

        if info["superseding_model"]:
            print(f"\n    ⚠️  Superseded by: {info['superseding_model']}")
        print()


# ─── Disk Versions Command ────────────────────────────────────────────────


def cmd_versions_disk(args, db: ModelRegistryDB):
    """List versioned checkpoints stored in _versions/ directory."""
    versions_dir = ROOT / "data" / "model_zoo" / "checkpoints" / "_versions"

    if not versions_dir.exists():
        print("  No _versions/ directory found (no models have been auto-versioned yet).")
        return

    pt_files = sorted(versions_dir.glob("*.pt"))
    if args.filter:
        pt_files = [f for f in pt_files if fnmatch.fnmatch(f.name, args.filter)]

    if not pt_files:
        print("  No versioned checkpoints found matching filter.")
        return

    if args.json:
        entries = []
        for f in pt_files:
            entry = {"filename": f.name, "size_mb": f.stat().st_size / 1024 / 1024}
            sidecar = f.with_suffix(".json")
            if sidecar.exists():
                entry["metadata"] = json.loads(sidecar.read_text())
            entries.append(entry)
        print(json.dumps(entries, indent=2))
        return

    print(f"\n  Versioned Checkpoints ({len(pt_files)} files in _versions/):\n")
    print(f"  {'Filename':<60} {'Size':<8} {'Pass%':<7} {'Pts':<6} {'Superseded By'}")
    print(f"  {'─' * 60} {'─' * 8} {'─' * 7} {'─' * 6} {'─' * 30}")

    for f in pt_files:
        size_mb = f.stat().st_size / 1024 / 1024
        pass_rate = "?"
        pts = "?"
        superseded = ""

        sidecar = f.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                pass_rate = f"{meta.get('pass_rate', 0):.0%}"
                pts = str(meta.get("n_training_points", "?"))
                superseded = meta.get("superseded_by", "")[:30]
            except (json.JSONDecodeError, KeyError):
                pass

        name_short = f.name[:58] + ".." if len(f.name) > 60 else f.name
        print(f"  {name_short:<60} {size_mb:>5.1f}MB {pass_rate:<7} {pts:<6} {superseded}")

    total_mb = sum(f.stat().st_size for f in pt_files) / 1024 / 1024
    print(f"\n  Total: {total_mb:.1f} MB in {len(pt_files)} versioned checkpoints\n")


# ─── Failure Diagnostic Commands  ─────────────────────────


def cmd_diagnose(args, db: ModelRegistryDB):
    """Run failure diagnostics for a specific model."""
    from dataclasses import asdict

    summary = db.run_failure_diagnostics(args.model_id, force=args.force)

    if summary is None:
        print(f"  ✗ Could not diagnose: {args.model_id}")
        print("    (Model not found or no dashboard data available)")
        return

    if args.json:
        print(json.dumps(asdict(summary), indent=2))
        return

    mode_icons = {
        "healthy": "✅",
        "gap_masking": "🎭",
        "contaminated_training": "☣️ ",
        "intrinsic_vqe_error": "🔧",
        "generalization_failure": "📉",
        "mixed": "🔀",
        "unknown": "❓",
    }
    icon = mode_icons.get(summary.primary_mode, "•")

    print(f"\n  {icon} Failure Diagnostic for {args.model_id}:\n")
    print(f"    Primary mode:  {summary.primary_mode}")
    print(f"    Confidence:    {summary.confidence:.0%}")

    if summary.secondary_modes:
        print(f"    Secondary:     {', '.join(summary.secondary_modes)}")

    if summary.explanation:
        print("\n    Explanation:")
        # Word-wrap at 70 chars
        words = summary.explanation.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                print(line)
                line = "    " + word
            else:
                line += " " + word if line.strip() else word
        if line.strip():
            print(line)

    # Key metrics
    print("\n    Key Metrics:")
    if summary.per_site_verified is not None:
        print(f"      Per-site verified:   {summary.per_site_verified:.4f}")
    if summary.gap_masked_fraction is not None:
        print(f"      Gap masked fraction: {summary.gap_masked_fraction:.0%}")
    if summary.contamination_severity:
        print(f"      Contamination:       {summary.contamination_severity}")
    if summary.h_range_overlap is not None:
        print(f"      H-range overlap:     {summary.h_range_overlap:.0%}")

    print(f"\n    Diagnosed at: {summary.diagnosed_at[:19]}")

    # Show auto-tags
    record = db.get_model(args.model_id)
    if record and record.tags:
        diag_tags = {"gap-masked", "contaminated", "ansatz-limited", "cross-n-degraded", "clean"}
        auto_tags = [t for t in record.tags if t in diag_tags]
        if auto_tags:
            print(f"    Auto-tags:    {', '.join(auto_tags)}")
    print()


def cmd_diagnostics(args, db: ModelRegistryDB):
    """Run failure diagnostics for all models."""
    results = db.run_all_failure_diagnostics(
        topology=args.topology,
        force=args.force,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("\n  Failure Diagnostics Summary:\n")
    print(f"    Diagnosed:   {results['n_diagnosed']}")
    print(f"    Skipped:     {results['n_skipped']} (already diagnosed)")
    print(f"    Failed:      {results['n_failed']}")

    if results["mode_distribution"]:
        print("\n    Mode Distribution:")
        for mode, count in sorted(results["mode_distribution"].items(), key=lambda x: -x[1]):
            mode_icons = {
                "healthy": "✅",
                "gap_masking": "🎭",
                "contaminated_training": "☣️ ",
                "intrinsic_vqe_error": "🔧",
                "generalization_failure": "📉",
                "mixed": "🔀",
                "unknown": "❓",
            }
            icon = mode_icons.get(mode, "•")
            print(f"      {icon} {mode}: {count}")
    print()


def cmd_comprehensive_health(args, db: ModelRegistryDB):
    """Show comprehensive health report for a model."""
    health = db.get_comprehensive_health(args.model_id)

    if args.json:
        print(json.dumps(health, indent=2))
        return

    if "error" in health:
        print(f"  ✗ {health['error']}")
        return

    status_icons = {
        "healthy": "✅",
        "warning": "⚠️ ",
        "critical": "❌",
    }
    rec_colors = {
        "deploy": "✅ DEPLOY",
        "investigate": "⚠️  INVESTIGATE",
        "retrain": "🔄 RETRAIN",
        "do_not_use": "🚫 DO NOT USE",
    }

    icon = status_icons.get(health["status"], "❓")
    rec = rec_colors.get(health["recommendation"], health["recommendation"])

    print(f"\n  {icon} Comprehensive Health Report for {health['model_id']}:\n")
    print(f"    Status:         {health['status'].upper()}")
    print(f"    Recommendation: {rec}")
    print(f"    Topology:       {health['topology']}")

    # Integrity section
    print("\n    ── Integrity ──")
    integ = health["integrity"]
    print(f"    Checkpoint:     {'✓' if integ['checkpoint_exists'] else '✗'}")
    print(f"    Hash:           {'✓' if integ['hash_matches'] else '✗'}")
    print(f"    Manifest:       {'✓' if integ['manifest_consistent'] else '✗'}")
    print(
        f"    Training data:  {'✓' if integ['training_data_exists'] else '✗'} "
        f"(verified: {integ['training_data_verified_ratio']:.0%})"
    )

    # Quality section
    print("\n    ── Quality ──")
    qual = health["quality"]
    print(f"    Training utility: {qual['training_utility'] or 'unknown'}")
    print(f"    Pass rate (dual): {qual['pass_rate_dual']:.0%}")
    print(f"    Needs retrain:    {'Yes' if qual['needs_retrain'] else 'No'}")
    print(f"    Model stale:      {'Yes' if qual['model_stale'] else 'No'}")

    # Diagnostics section
    print("\n    ── Diagnostics ──")
    diag = health["diagnostics"]
    mode_icons = {
        "healthy": "✅",
        "gap_masking": "🎭",
        "contaminated_training": "☣️ ",
        "intrinsic_vqe_error": "🔧",
        "generalization_failure": "📉",
        "mixed": "🔀",
        "unknown": "❓",
    }
    mode_icon = mode_icons.get(diag["primary_mode"], "•")
    print(f"    Primary mode:   {mode_icon} {diag['primary_mode']} ({diag['confidence']:.0%})")
    if diag["secondary_modes"]:
        print(f"    Secondary:      {', '.join(diag['secondary_modes'])}")

    # Issues section
    if health["issues"]:
        print(f"\n    ── Issues ({len(health['issues'])}) ──")
        for issue in health["issues"]:
            print(f"      • {issue}")

    # Tags
    if health["tags"]:
        print(f"\n    Tags: {', '.join(health['tags'])}")
    print()


def cmd_health_dashboard(args, db: ModelRegistryDB):
    """Generate comprehensive health dashboard."""
    dashboard = db.generate_health_dashboard()

    if args.json:
        print(json.dumps(dashboard, indent=2))
        return

    summary = dashboard["summary"]

    print(f"\n  ╔{'═' * 52}╗")
    print(f"  ║  Model Health Dashboard{'':>27}║")
    print(f"  ║  Generated: {dashboard['generated_at'][:19]:<29}║")
    print(f"  ╠{'═' * 52}╣")
    print(f"  ║  Total active:    {summary['total_active']:<32}║")
    print(f"  ║  Healthy:         {summary['healthy']:<32}✅ ║")
    print(f"  ║  Warning:         {summary['warning']:<32}⚠️  ║")
    print(f"  ║  Critical:        {summary['critical']:<32}❌ ║")
    print(f"  ╚{'═' * 52}╝\n")

    # By failure mode
    by_mode = dashboard.get("by_failure_mode", {})
    if by_mode:
        print("  Failure Mode Distribution:")
        mode_icons = {
            "healthy": "✅",
            "gap_masking": "🎭",
            "contaminated_training": "☣️ ",
            "intrinsic_vqe_error": "🔧",
            "generalization_failure": "📉",
            "mixed": "🔀",
            "unknown": "❓",
        }
        for mode, model_ids in sorted(by_mode.items(), key=lambda x: -len(x[1])):
            icon = mode_icons.get(mode, "•")
            print(f"    {icon} {mode}: {len(model_ids)} model(s)")
        print()

    # Action items
    action_items = dashboard.get("action_items", [])
    if action_items:
        print(f"  Action Items ({len(action_items)}):\n")
        for item in action_items[:10]:  # Top 10
            priority_icon = "🔴" if item["priority"] == "high" else "🟡"
            print(f"    {priority_icon} {item['model_id'][:45]}")
            print(f"       Topology: {item['topology']} | Action: {item['recommendation']}")
            if item["issues"]:
                print(f"       Issues: {item['issues'][0][:60]}")
            print()

        if len(action_items) > 10:
            print(f"    ... and {len(action_items) - 10} more")
    else:
        print("  ✅ No action items — all models healthy!")
    print()


# ─── Data Consistency Command ────────────────────────────────────────────────


def cmd_consistency(args, db: ModelRegistryDB):
    """Run full data consistency validation across all sources."""
    import json as _json

    from qmbp_simulation.analysis.metrics import validate_data_consistency

    result = validate_data_consistency(verbose=not args.json)

    if args.json:
        output = {
            "is_consistent": result["is_consistent"],
            "n_checks": result["n_checks"],
            "n_issues": result["n_issues"],
            "findings": result["findings"],
            "zoo_vs_comparison": result["zoo_vs_comparison"],
            "registry_vs_curves": result["registry_vs_curves"],
            "cross_n_selection_issues": result["cross_n_selection_issues"],
        }
        print(_json.dumps(output, indent=2, default=str))
        return

    # Additional: GT coherence
    from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

    gt = validate_gt_npz_coherence()
    if gt["n_files_with_issues"] > 0:
        print(
            f"\n  🔴 GT↔NPZ: {gt['n_files_with_issues']} files with stale e_exact "
            f"(max_delta={gt['max_delta']:.2e})"
        )
    else:
        print(f"\n  ✅ GT↔NPZ: coherent ({gt['n_points_checked']} points)")

    # Zoo vs Comparison summary
    zoo_comp = result["zoo_vs_comparison"]
    if zoo_comp:
        print(f"\n  Zoo ↔ Comparison ({len(zoo_comp)} models checked):")
        for ckpt, info in sorted(zoo_comp.items(), key=lambda x: -x[1]["delta"]):
            icon = "✅" if info["consistent"] else "⚠️"
            print(
                f"    {icon} {ckpt[:45]}: zoo={info['zoo_pass_rate']:.0%} "
                f"comp={info['comparison_avg']:.0%} (Δ={info['delta']:.0%})"
            )
            if info["comparison_by_n"]:
                by_n_str = ", ".join(
                    f"N{n}={r:.0%}" for n, r in sorted(info["comparison_by_n"].items())
                )
                print(f"         by_N: {by_n_str}")

    # Cross-N selection issues
    xn_issues = result["cross_n_selection_issues"]
    if xn_issues:
        print(f"\n  Cross-N Selection Issues ({len(xn_issues)}):")
        for iss in xn_issues:
            print(f"    ⚠️ {iss['topology']} N={iss['n_target']}: {iss['issue']}")


# ─── MT vs ST Comparison Command ─────────────────────────────────────────────


def cmd_compare(args, db: ModelRegistryDB):
    """Compare MT vs ST models with full dashboard integration.

    Reads comparison results from model_comparison/ JSONs and cross-references
    with the dashboard NPZ data for a unified view of model performance.
    """
    import json as _json
    from pathlib import Path as _Path

    from qmbp_simulation.analysis.evaluation_report import generate_mt_vs_st_table

    # Build filter kwargs
    kwargs = {}
    if args.topology:
        kwargs["topology_filter"] = args.topology
    if args.n_min:
        kwargs["n_min"] = args.n_min
    if args.n_max:
        kwargs["n_max"] = args.n_max

    lines, summary = generate_mt_vs_st_table(latest_only=not args.all_runs, **kwargs)

    if args.json:
        print(_json.dumps(summary, indent=2, default=str))
        return

    total = summary.get("total", 0)
    if total == 0:
        print("\n  ⚠️ No comparison data found.")
        print("  Run model comparisons first:")
        print(
            "    .venv/bin/python scripts/experiment_runners/cross_topology/"
            "run_model_comparison.py --topology chain_1d --target-n 10 16 20 --auto-detect"
        )
        return

    # Global summary
    mt_wins = summary["mt_wins"]
    st_wins = summary["st_wins"]
    ties = summary["ties"]
    mt_avg = summary.get("mt_avg_pass_rate", 0.0)
    st_avg = summary.get("st_avg_pass_rate", 0.0)

    global_winner = "MT" if mt_avg > st_avg + 0.01 else ("ST" if st_avg > mt_avg + 0.01 else "Tie")
    winner_icon = "🟢" if global_winner == "MT" else ("🔴" if global_winner == "ST" else "⚪")

    print(f"\n  ╔{'═' * 56}╗")
    print(f"  ║  MT vs ST Model Comparison{'':>28}║")
    print(f"  ║  Generated: {summary.get('generated_at', '?')[:19]:<33}║")
    print(f"  ╠{'═' * 56}╣")
    print(f"  ║  Score: MT {mt_wins} — ST {st_wins} — Ties {ties}{'':<25}║")
    print(f"  ║  MT avg pass_rate: {mt_avg:.0%} | ST avg: {st_avg:.0%}{'':<16}║")
    print(f"  ║  Overall winner: {winner_icon} {global_winner:<34}║")
    print(f"  ╚{'═' * 56}╝\n")

    # Per-topology breakdown
    per_topology = summary.get("per_topology", {})
    if per_topology:
        print(f"  {'Topology':<14} {'MT pass%':>9} {'ST pass%':>9} {'Winner':>8} {'Δ':>7} {'MT W':>5} {'ST W':>5}")
        print(f"  {'─' * 14} {'─' * 9} {'─' * 9} {'─' * 8} {'─' * 7} {'─' * 5} {'─' * 5}")
        for topo in sorted(per_topology.keys()):
            info = per_topology[topo]
            icon = "🟢" if info["winner"] == "MT" else ("🔴" if info["winner"] == "ST" else "⚪")
            print(
                f"  {topo:<14} {info['mt_avg_pass_rate']:>8.0%} "
                f"{info['st_avg_pass_rate']:>9.0%} "
                f"{icon} {info['winner']:<5} "
                f"{info['delta']:>+6.0%} "
                f"{info['mt_wins']:>5} {info['st_wins']:>5}"
            )
        print()

    # Per-scenario detail (if verbose)
    if args.verbose:
        per_scenario = summary.get("per_scenario", [])
        if per_scenario:
            print(f"  {'Topology':<14} {'N':>3} {'MT%':>5} {'MT gr':>5} {'ST%':>5} {'ST gr':>5} {'Win':>4}")
            print(f"  {'─' * 14} {'─' * 3} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 4}")
            for s in per_scenario:
                icon = "✅" if s["winner"] == "MT" else ("❌" if s["winner"] == "ST" else "—")
                print(
                    f"  {s['topology']:<14} {s['n_qubits']:>3} "
                    f"{s['mt_pass_rate']:>4.0%} {s['mt_grade']:>5} "
                    f"{s['st_pass_rate']:>4.0%} {s['st_grade']:>5} "
                    f"{icon:>4}"
                )
            print()

    # Cross-reference with dashboard NPZ data for enriched context
    if args.enrich:
        _enrich_with_dashboard(per_topology, summary)

    # Save report if requested
    if args.save:
        out_path = _Path(__file__).resolve().parents[2] / "results" / "mt_vs_st_report.md"
        generate_mt_vs_st_table(output_path=out_path, latest_only=not args.all_runs, **kwargs)
        print(f"  📄 Report saved: {out_path.relative_to(out_path.parents[2])}")


def _enrich_with_dashboard(per_topology: dict, summary: dict):
    """Cross-reference MT vs ST comparison with dashboard NPZ quality data."""
    import json as _json
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[2]
    dash_path = _ROOT / "data" / "model_quality_dashboard.json"
    if not dash_path.exists():
        print("  ⚠️ Dashboard not found — skipping enrichment.")
        return

    with open(dash_path) as f:
        dashboard = _json.load(f)

    configs = dashboard.get("configs", [])

    # Group NPZ configs by topology
    npz_by_topo: dict[str, list[dict]] = {}
    for c in configs:
        npz_by_topo.setdefault(c["topology"], []).append(c)

    print("  ┌─ Dashboard Enrichment (NPZ training data quality) ─────────────────┐")
    for topo, info in sorted(per_topology.items()):
        topo_configs = npz_by_topo.get(topo, [])
        if not topo_configs:
            continue

        # Aggregate training data quality for this topology
        total_pts = sum(c.get("n_points", 0) for c in topo_configs)
        n_useful = sum(1 for c in topo_configs if c.get("training_utility") == "useful")
        avg_de_gap = (
            sum(c.get("mean_de_gap", 0) for c in topo_configs) / max(len(topo_configs), 1)
        )
        best_pass_dual = max(
            (c.get("pass_rate_dual_criterion", 0) for c in topo_configs), default=0
        )
        n_values = sorted(set(c["n_qubits"] for c in topo_configs))

        # Check if MT or ST models are stale for this topology
        zoo_pass = max(
            (c.get("zoo_pass_rate", 0) or 0 for c in topo_configs), default=0
        )
        is_stale = any(c.get("model_stale") for c in topo_configs)

        status = "✅" if n_useful == len(topo_configs) else "⚠️"
        stale_str = " (⚠️ stale model)" if is_stale else ""

        print(
            f"  │ {topo:<12} {total_pts:>4} pts, "
            f"{n_useful}/{len(topo_configs)} useful, "
            f"best_dual={best_pass_dual:.0%}, "
            f"N={n_values}{stale_str} {status}"
        )

    # MT model info from dashboard
    mt_info = dashboard.get("multi_topology_models", {})
    if mt_info.get("n_models", 0) > 0:
        best_mt = mt_info["best_pass_rate"]
        n_mt_pts = sum(m.get("n_training_points", 0) for m in mt_info.get("models", []))
        print(f"  │")
        print(f"  │ MT model: pass_rate={best_mt:.0%}, training_pts={n_mt_pts}")

    # Dashboard comparison section (if already embedded)
    dash_compare = dashboard.get("mt_vs_st_comparison", {})
    if dash_compare:
        dash_global = dash_compare.get("global", {})
        print(f"  │")
        print(
            f"  │ Dashboard embedded: "
            f"MT {dash_global.get('mt_wins', 0)} — "
            f"ST {dash_global.get('st_wins', 0)} — "
            f"Ties {dash_global.get('ties', 0)}"
        )

    print("  └──────────────────────────────────────────────────────────────────────┘")
    print()


# ─── Helpers ────────────────────────────────────────────────────────────────


def _event_icon(event_type: str) -> str:
    icons = {
        "registered": "🆕",
        "retrained": "🔄",
        "evaluated": "📊",
        "superseded": "⏭️ ",
        "archived": "📦",
        "pass_rate_updated": "📈",
        "regression_detected": "⚠️ ",
        "tag_added": "🏷️ ",
        "tag_removed": "🏷️ ",
        "integrity_checked": "🔍",
        "failure_diagnosed": "🩺",
        "auto_versioned": "💾",
        "training_data_changed": "📝",
        "auto_retrain_triggered": "🔁",
        "quality_degraded": "📉",
        "needs_retrain_flagged": "🚩",
        "needs_retrain_cleared": "✅",
        "dashboard_synced": "📋",
    }
    return icons.get(event_type, "•")


def _print_model_card(r, verbose: bool = False):
    """Print a formatted model card."""
    n_str = "+".join(str(n) for n in r.training.n_values_used) if r.training.n_values_used else "—"
    ppn = (
        ", ".join(f"N{k}={v}" for k, v in r.training.points_per_n.items())
        if r.training.points_per_n
        else "—"
    )

    print(f"  ┌─ {r.model_id}")
    print(f"  │  Topology:    {r.topology}")
    print(f"  │  Model:       {r.model_name} ({r.architecture})")
    print(f"  │  N values:    [{n_str}]")
    print(f"  │  Total pts:   {r.training.total_training_points}")
    print(f"  │  Per-N:       {ppn}")
    print(f"  │  h range:     {r.training.h_range}")
    print(f"  │  Created:     {r.created}  ({r.runner_tag}/{r.date_tag})")
    print(f"  │  Status:      {r.status}")

    # Version
    version_str = f"v{r.version}"
    if r.version_history:
        version_str += f" ({len(r.version_history)} previous)"
    print(f"  │  Version:     {version_str}")

    # Tags
    if r.tags:
        print(f"  │  Tags:        {', '.join(r.tags)}")

    # Training quality score from zoo
    try:
        from qmbp_simulation.predictors.model_zoo import compute_training_quality_score

        score = compute_training_quality_score(
            r.topology,
            n_qubits=0
            if len(r.training.n_values_used or []) > 1
            else (r.training.n_values_used[0] if r.training.n_values_used else 0),
            p_layers=1,
            model=r.model_name,
        )
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  │  Quality:     {bar} {score:.3f}")
    except Exception:
        pass

    if r.evaluations:
        latest = r.evaluations[-1]
        print(f"  │  Evaluations: {len(r.evaluations)} total")
        print(
            f"  │  Last eval:   pass_rate={latest.pass_rate_dual:.0%} ΔE/gap={latest.mean_de_gap:.4f}"
        )
        if latest.target_n_values:
            print(f"  │               target_N={latest.target_n_values}")

    if verbose and r.evaluations:
        print("  │  ── Evaluation History ──")
        for i, ev in enumerate(r.evaluations):
            ts = ev.evaluated_at[:19] if ev.evaluated_at else "?"
            print(
                f"  │   [{i + 1}] {ts}  pass={ev.pass_rate_dual:.0%}  ΔE/gap={ev.mean_de_gap:.4f}  N={ev.target_n_values}"
            )

    # Training metrics (#1)
    if verbose and r.training.training_metrics.epochs > 0:
        tm = r.training.training_metrics
        print("  │  ── Training Metrics ──")
        print(f"  │   Epochs:     {tm.epochs} (best: {tm.best_epoch})")
        print(f"  │   Final loss: {tm.final_loss:.4e}")
        print(f"  │   Final MSE:  {tm.final_mse:.4e}")
        print(f"  │   Time:       {tm.training_time_seconds:.1f}s")
        print(f"  │   Status:     {tm.convergence_status}")

    # Dashboard quality
    if verbose and r.dashboard_quality.training_utility:
        dq = r.dashboard_quality
        print("  │  ── Dashboard Quality ──")
        print(f"  │   Utility:    {dq.training_utility}")
        print(f"  │   Needs retrain: {'Yes' if dq.needs_retrain else 'No'}")
        print(f"  │   Synced:     {dq.last_synced[:19] if dq.last_synced else 'never'}")

        # Failure diagnostic
        fd = dq.failure_diagnostic
        if fd.primary_mode:
            mode_icons = {
                "healthy": "✅",
                "gap_masking": "🎭",
                "contaminated_training": "☣️ ",
                "intrinsic_vqe_error": "🔧",
                "generalization_failure": "📉",
                "mixed": "🔀",
                "unknown": "❓",
            }
            icon = mode_icons.get(fd.primary_mode, "•")
            print("  │  ── Failure Diagnostic ──")
            print(f"  │   Mode:       {icon} {fd.primary_mode} ({fd.confidence:.0%} conf)")
            if fd.secondary_modes:
                print(f"  │   Secondary:  {', '.join(fd.secondary_modes)}")
            if fd.explanation:
                # Wrap explanation to 60 chars
                exp = fd.explanation[:80] + "..." if len(fd.explanation) > 80 else fd.explanation
                print(f"  │   Reason:     {exp}")

    if r.superseded_by:
        print(f"  │  Superseded:  → {r.superseded_by}")
    if r.notes:
        print(f"  │  Notes:       {r.notes[:80]}")
    print(f"  └{'─' * 60}")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Model Registry DB — query, inspect, and audit trained MPNN models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    p_list = subparsers.add_parser("list", help="List models with filters")
    p_list.add_argument("--topology", "-t", help="Filter by topology")
    p_list.add_argument("--model-name", "-m", help="Filter by model name")
    p_list.add_argument("--model-id", help="Filter by model ID (supports * wildcard)")
    p_list.add_argument("--min-points", type=int, help="Minimum training points")
    p_list.add_argument("--min-n-values", type=int, help="Minimum N values used")
    p_list.add_argument("--include-archived", action="store_true")

    # get
    p_get = subparsers.add_parser("get", help="Get detailed model info")
    p_get.add_argument("model_id", help="Model ID (supports * wildcard)")

    # summary
    subparsers.add_parser("summary", help="Show registry statistics")

    # history
    p_hist = subparsers.add_parser("history", help="Show event history")
    p_hist.add_argument("--model-id", help="Filter by model ID (supports * wildcard)")
    p_hist.add_argument("--event-type", "-e", help="Filter by event type")
    p_hist.add_argument("--topology", "-t", help="Filter by topology")
    p_hist.add_argument("--limit", "-n", type=int, default=50, help="Max events to show")

    # regressions
    p_reg = subparsers.add_parser("regressions", help="Detect regressions")
    p_reg.add_argument(
        "--threshold", type=float, default=0.05, help="Regression threshold (default: 0.05)"
    )

    # timeline
    p_tl = subparsers.add_parser("timeline", help="Chronological timeline for a model")
    p_tl.add_argument("model_id", help="Model ID (exact)")

    # sync
    subparsers.add_parser("sync", help="Sync from manifest + NPZ + dashboard enrichment")

    # tag
    p_tag = subparsers.add_parser("tag", help="Add or remove tags from a model")
    p_tag.add_argument("model_id", help="Model ID")
    p_tag.add_argument("--add", "-a", help="Tag to add")
    p_tag.add_argument("--remove", "-r", help="Tag to remove")
    p_tag.add_argument("--list", dest="list_tags", action="store_true", help="List tags for model")

    # tags
    p_tags = subparsers.add_parser("tags", help="List all tags or query by tag")
    p_tags.add_argument("--query", "-q", help="Find models with this tag")
    p_tags.add_argument("--include-archived", action="store_true")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate model integrity")
    p_val.add_argument("model_id", nargs="?", help="Model ID (or validate all if omitted)")

    # health
    p_health = subparsers.add_parser("health", help="Show training data health for a model")
    p_health.add_argument("model_id", help="Model ID")

    # best
    p_best = subparsers.add_parser("best", help="Find the best model for deployment")
    p_best.add_argument("--topology", "-t", required=True, help="Target topology")
    p_best.add_argument("--model-name", "-m", default="tfim_bond_resolved", help="Model name")
    p_best.add_argument("--p-layers", "-p", type=int, default=1, help="HVA depth")
    p_best.add_argument("--n-target", "-n", type=int, required=True, help="Target N for prediction")
    p_best.add_argument("--require-tag", help="Only consider models with this tag")

    # versions
    p_versions = subparsers.add_parser("versions", help="List models with version info")
    p_versions.add_argument("--topology", "-t", help="Filter by topology")
    p_versions.add_argument("--model-name", "-m", help="Filter by model name")
    p_versions.add_argument("--p-layers", "-p", type=int, help="Filter by p_layers")

    # versions-disk (lists physical _versions/ directory)
    p_vdisk = subparsers.add_parser("versions-disk", help="List versioned checkpoints on disk")
    p_vdisk.add_argument("--filter", "-f", help="Glob filter for filenames (e.g. '*chain_1d*')")

    # version
    p_version = subparsers.add_parser("version", help="Show version info for a model")
    p_version.add_argument("model_id", help="Model ID")
    p_version.add_argument("--chain", "-c", action="store_true", help="Show full version chain")

    # diagnose
    p_diag = subparsers.add_parser("diagnose", help="Run failure diagnostics for a model")
    p_diag.add_argument("model_id", help="Model ID")
    p_diag.add_argument(
        "--force", "-f", action="store_true", help="Re-run even if already diagnosed"
    )

    # diagnostics
    p_diags = subparsers.add_parser("diagnostics", help="Run failure diagnostics for all models")
    p_diags.add_argument("--topology", "-t", help="Filter by topology")
    p_diags.add_argument("--force", "-f", action="store_true", help="Re-run all (force)")

    # comprehensive-health
    p_comp = subparsers.add_parser("comprehensive-health", help="Full health report for a model")
    p_comp.add_argument("model_id", help="Model ID")

    # health-dashboard
    subparsers.add_parser("health-dashboard", help="Generate health dashboard for all models")

    # compare (MT vs ST)
    p_compare = subparsers.add_parser(
        "compare", help="MT vs ST model comparison with dashboard integration"
    )
    p_compare.add_argument("--topology", "-t", nargs="*", help="Filter by topology (one or more)")
    p_compare.add_argument("--n-min", type=int, help="Minimum N value")
    p_compare.add_argument("--n-max", type=int, help="Maximum N value")
    p_compare.add_argument("--all-runs", action="store_true",
                           help="Use all comparison runs (not just latest per topology)")
    p_compare.add_argument("--enrich", action="store_true", default=True,
                           help="Cross-reference with dashboard NPZ quality data")
    p_compare.add_argument("--no-enrich", dest="enrich", action="store_false")
    p_compare.add_argument("--save", action="store_true",
                           help="Save markdown report to results/mt_vs_st_report.md")
    p_compare.add_argument("-v", "--verbose", action="store_true",
                           help="Show per-N breakdown")

    # consistency (cross-source data validation)
    subparsers.add_parser(
        "consistency",
        help="Validate data consistency across zoo, dashboard, comparisons, registry",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = ModelRegistryDB()

    # Dispatch
    dispatch = {
        "list": cmd_list,
        "get": cmd_get,
        "summary": cmd_summary,
        "history": cmd_history,
        "regressions": cmd_regressions,
        "timeline": cmd_timeline,
        "sync": cmd_sync,
        "tag": cmd_tag,
        "tags": cmd_tags,
        "validate": cmd_validate,
        "health": cmd_health,
        "best": cmd_best,
        "versions": cmd_versions,
        "versions-disk": cmd_versions_disk,
        "version": cmd_version,
        "diagnose": cmd_diagnose,
        "diagnostics": cmd_diagnostics,
        "comprehensive-health": cmd_comprehensive_health,
        "health-dashboard": cmd_health_dashboard,
        "compare": cmd_compare,
        "consistency": cmd_consistency,
    }
    dispatch[args.command](args, db)


if __name__ == "__main__":
    main()
