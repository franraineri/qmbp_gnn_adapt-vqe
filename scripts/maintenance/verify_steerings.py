#!/usr/bin/env python3
"""Verify steering files: pattern activation, quality, staleness, and split suggestions.

Checks that .kiro/steering/*.md files are correct, well-maintained, and
optimally structured for AI agent context efficiency.

Usage:
    # Full check (all steerings)
    python scripts/maintenance/verify_steerings.py

    # Check specific file(s)
    python scripts/maintenance/verify_steerings.py .kiro/steering/accelerated-pipeline.md

    # Check a folder
    python scripts/maintenance/verify_steerings.py .kiro/steering/

    # Only run specific checks
    python scripts/maintenance/verify_steerings.py --check activation quality

    # Include staleness analysis (uses git log, slower)
    python scripts/maintenance/verify_steerings.py --check staleness

    # Show split suggestions (heuristic analysis)
    python scripts/maintenance/verify_steerings.py --check split
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEERING_DIR = PROJECT_ROOT / ".kiro" / "steering"

# Days since last touch after which a file is considered stale
STALE_DAYS_THRESHOLD = 60


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def parse_front_matter(filepath: Path) -> dict[str, str]:
    """Extract YAML front-matter from a markdown file."""
    text = filepath.read_text()
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    fm = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip().strip('"').strip("'")
    return fm


def get_body(filepath: Path) -> str:
    """Get steering body content (after front-matter)."""
    text = filepath.read_text()
    match = re.match(r"^---\s*\n.*?\n---\s*\n?", text, re.DOTALL)
    if match:
        return text[match.end():]
    return text


def glob_match(filepath: str, pattern: str) -> bool:
    """Check if filepath matches a glob pattern (supporting **)."""
    pattern = pattern.strip()
    if "**" not in pattern:
        return fnmatch.fnmatch(filepath, pattern)
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return filepath.startswith(prefix + "/") or filepath == prefix
    parts = pattern.split("**/")
    if len(parts) == 2:
        prefix, suffix = parts
        if prefix and not filepath.startswith(prefix):
            return False
        remaining = filepath[len(prefix):] if prefix else filepath
        segments = remaining.split("/")
        for i in range(len(segments)):
            candidate = "/".join(segments[i:])
            if fnmatch.fnmatch(candidate, suffix):
                return True
    return False


def matches_any(filepath: str, patterns: list[str]) -> bool:
    """Check if filepath matches any of the glob patterns."""
    return any(glob_match(filepath, p) for p in patterns)


def scan_steerings(
    targets: list[Path] | None = None,
) -> dict[str, list[str]]:
    """Scan steering files and extract their fileMatchPatterns.

    Parameters
    ----------
    targets : list[Path] | None
        If provided, only scan these files. Otherwise scan all in STEERING_DIR.
    """
    files = targets if targets else sorted(STEERING_DIR.glob("*.md"))
    result = {}
    for md_file in files:
        if not md_file.exists() or md_file.is_dir():
            continue
        fm = parse_front_matter(md_file)
        inclusion = fm.get("inclusion", "always")
        if inclusion == "fileMatch":
            pattern_str = fm.get("fileMatchPattern", "")
            if pattern_str:
                patterns = [p.strip() for p in pattern_str.split(",")]
                result[md_file.stem] = patterns
    return result


def get_git_last_touched(filepath: Path) -> datetime | None:
    """Get the last git commit date that touched a file."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(filepath)],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return datetime.fromisoformat(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def find_matched_files(patterns: list[str], limit: int = 100) -> list[Path]:
    """Find real project files matching a set of glob patterns."""
    import glob as _glob

    matched: list[Path] = []
    for pat in patterns:
        full_pattern = str(PROJECT_ROOT / pat)
        hits = _glob.glob(full_pattern, recursive=True)
        matched.extend(Path(h) for h in hits[:limit])
    return matched[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# Expected Activations (key files → required steerings)
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_ACTIVATIONS: dict[str, list[str]] = {
    "scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py": [
        "runner-standards", "noiseless-runner-patterns",
        "eval-cache-guidelines", "accelerated-pipeline",
        "reuse-existing-infrastructure",
    ],
    "scripts/experiment_runners/noiseless/run_noiseless_pipeline.py": [
        "runner-standards", "noiseless-runner-patterns",
        "eval-cache-guidelines", "reuse-existing-infrastructure",
    ],
    "src/qmbp_simulation/framework/runner_base.py": [
        "runner-standards", "code-style", "reuse-existing-infrastructure",
    ],
    "src/qmbp_simulation/analysis/metrics.py": [
        "code-style", "reuse-existing-infrastructure",
    ],
    "src/qmbp_simulation/execution/eval_cache.py": [
        "eval-cache-guidelines", "code-style", "reuse-existing-infrastructure",
    ],
    "src/qmbp_simulation/pipeline/accelerated.py": [
        "accelerated-pipeline", "code-style",
        "eval-cache-guidelines", "reuse-existing-infrastructure",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Checks
# ═══════════════════════════════════════════════════════════════════════════════


def check_activation(steerings: dict[str, list[str]]) -> list[str]:
    """Check 1: Verify expected pattern activations."""
    issues = []
    print("─── Check 1: Pattern Activation ───")
    for filepath, expected in EXPECTED_ACTIVATIONS.items():
        print(f"  📄 {filepath}")
        for steer_name in expected:
            if steer_name not in steerings:
                print(f"    ⚠ {steer_name}: not found (may be always-included)")
                continue
            ok = matches_any(filepath, steerings[steer_name])
            if not ok:
                issues.append(f"MISS: {filepath} should trigger {steer_name}")
                print(f"    ✗ {steer_name} — MISS!")
            else:
                print(f"    ✓ {steer_name}")
        print()
    return issues


def check_quality(targets: list[Path] | None = None) -> list[str]:
    """Check 2: Steering file quality (structure, references, content)."""
    issues = []
    files = targets if targets else sorted(STEERING_DIR.glob("*.md"))
    print("─── Check 2: Steering Quality ───")
    for md_file in files:
        if not md_file.exists():
            continue
        text = md_file.read_text()
        name = md_file.stem
        problems = []

        body = get_body(md_file)
        if len(body.strip()) < 50:
            problems.append("near-empty body (<50 chars)")

        fm = parse_front_matter(md_file)
        if text.startswith("---") and "inclusion" not in fm:
            problems.append("has front-matter but no 'inclusion' key")

        if fm.get("inclusion") == "fileMatch":
            pat = fm.get("fileMatchPattern", "")
            if not pat:
                problems.append("fileMatch mode but no fileMatchPattern")

        refs = re.findall(r'#\[\[file:(.*?)\]\]', text)
        for ref in refs:
            if not (PROJECT_ROOT / ref).exists():
                problems.append(f"broken ref: #[[file:{ref}]]")

        if not re.search(r'^#\s+\S', body, re.MULTILINE):
            problems.append("no H1 heading")

        if problems:
            print(f"  ✗ {name}.md:")
            for p in problems:
                print(f"      • {p}")
                issues.append(f"QUALITY: {name}.md — {p}")
        else:
            print(f"  ✓ {name}.md")
    print()
    return issues


def check_overlaps(steerings: dict[str, list[str]]) -> list[str]:
    """Check 3: Detect shared patterns between steerings."""
    issues = []
    print("─── Check 3: Overlap Detection ───")
    pattern_to_names: dict[str, list[str]] = defaultdict(list)
    for name, pats in steerings.items():
        for p in pats:
            pattern_to_names[p.strip()].append(name)

    found = False
    for pat, names in sorted(pattern_to_names.items()):
        if len(names) > 1:
            print(f"  ⚠ '{pat}' shared by: {', '.join(names)}")
            found = True
    if not found:
        print("  ✓ No pattern overlaps")
    print()
    return issues


def check_orphans(steerings: dict[str, list[str]]) -> list[str]:
    """Check 4: Patterns matching zero existing files."""
    import glob as _glob

    issues = []
    print("─── Check 4: Orphan Patterns ───")
    n_orphan = 0
    for name, pats in steerings.items():
        for pat in pats:
            if "**" not in pat:
                matches = _glob.glob(str(PROJECT_ROOT / pat))
                if not matches:
                    print(f"  ⚠ {name}: '{pat}' → 0 files")
                    issues.append(f"ORPHAN: {name} pattern '{pat}' matches nothing")
                    n_orphan += 1
    if n_orphan == 0:
        print("  ✓ All non-recursive patterns match existing files")
    print()
    return issues


def check_staleness(steerings: dict[str, list[str]]) -> list[str]:
    """Check 5: Identify stale patterns — files not touched in a while.

    For each steering's patterns, finds matched files and checks their
    last git commit date. If ALL matched files are older than STALE_DAYS_THRESHOLD,
    the steering might be outdated or covering dead code.
    """
    issues = []
    print(f"─── Check 5: Staleness (>{STALE_DAYS_THRESHOLD} days untouched) ───")
    now = datetime.now(timezone.utc)

    for name, pats in sorted(steerings.items()):
        matched_files = find_matched_files(pats, limit=20)
        if not matched_files:
            continue  # Already caught by orphan check

        # Get last-touched dates for matched files
        dates: list[tuple[Path, datetime | None]] = []
        for f in matched_files:
            if f.is_file() and not str(f).endswith(".pyc"):
                dt = get_git_last_touched(f)
                if dt:
                    dates.append((f, dt))

        if not dates:
            continue

        # Find the MOST RECENT touch across all matched files
        most_recent = max(dates, key=lambda x: x[1])  # type: ignore
        days_ago = (now - most_recent[1]).days  # type: ignore

        if days_ago > STALE_DAYS_THRESHOLD:
            print(
                f"  ⚠ {name}: last touched {days_ago}d ago "
                f"({most_recent[0].name}, {most_recent[1].strftime('%Y-%m-%d')})"  # type: ignore
            )
            issues.append(
                f"STALE: {name} — all matched files untouched for {days_ago} days"
            )
        else:
            print(f"  ✓ {name}: active ({days_ago}d ago)")

    print()
    return issues


def check_split_candidates(targets: list[Path] | None = None) -> list[str]:
    """Check 6: Suggest steerings that could benefit from splitting.

    Heuristics for split recommendation:
    1. Large body (>3000 chars) with multiple H2 sections covering different domains
    2. Pattern list covers very different file types (src + scripts + docs)
    3. Has >5 distinct topic clusters (detected via H2 headings)

    Split improves agent context efficiency: smaller, focused steerings
    only load when truly relevant, reducing noise in the context window.
    """
    issues = []
    print("─── Check 6: Split Suggestions ───")
    files = targets if targets else sorted(STEERING_DIR.glob("*.md"))

    for md_file in files:
        if not md_file.exists():
            continue
        name = md_file.stem
        body = get_body(md_file)
        fm = parse_front_matter(md_file)
        suggestions = []

        # Heuristic 1: Large file with many H2 sections
        h2_headings = re.findall(r'^##\s+(.+)$', body, re.MULTILINE)
        body_len = len(body)

        if body_len > 3000 and len(h2_headings) > 5:
            suggestions.append(
                f"large ({body_len} chars) with {len(h2_headings)} sections — "
                f"consider splitting by topic"
            )

        # Heuristic 2: Pattern covers very different directories
        if fm.get("inclusion") == "fileMatch":
            pat_str = fm.get("fileMatchPattern", "")
            pats = [p.strip() for p in pat_str.split(",")]
            # Identify distinct "domains" from pattern prefixes
            domains = set()
            for p in pats:
                if p.startswith("src/"):
                    domains.add("src")
                elif p.startswith("scripts/"):
                    domains.add("scripts")
                elif p.startswith("tests/"):
                    domains.add("tests")
                elif p.startswith("project_health/"):
                    domains.add("project_health")
                elif p.startswith("documentation/") or p.startswith("internal/"):
                    domains.add("docs")
                elif p.startswith("**/"):
                    domains.add("global")
            # If a steering covers 3+ distinct domains AND is large, suggest split
            if len(domains) >= 3 and body_len > 2000:
                suggestions.append(
                    f"covers {len(domains)} domains ({', '.join(sorted(domains))}) — "
                    f"consider domain-specific steerings"
                )

        # Heuristic 3: Multiple distinct code-block languages (mixing concerns)
        code_langs = set(re.findall(r'^```(\w+)', body, re.MULTILINE))
        code_langs.discard("")
        if len(code_langs) >= 3 and body_len > 2500:
            suggestions.append(
                f"mixes {len(code_langs)} code languages ({', '.join(sorted(code_langs))}) — "
                f"may conflate different usage patterns"
            )

        # Heuristic 4: Always-included but very large (wastes context on unrelated tasks)
        inclusion = fm.get("inclusion", "always")
        if inclusion == "always" and body_len > 4000:
            suggestions.append(
                f"always-included at {body_len} chars — "
                f"consider converting to fileMatch to reduce context noise"
            )

        if suggestions:
            print(f"  💡 {name}.md:")
            for s in suggestions:
                print(f"      • {s}")
            issues.append(f"SPLIT: {name}.md could be split")
        else:
            # Only print passing for verbose output
            pass

    if not issues:
        print("  ✓ No split suggestions")
    print()
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# CLI and Main
# ═══════════════════════════════════════════════════════════════════════════════

ALL_CHECKS = ["activation", "quality", "overlaps", "orphans", "staleness", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify steering files: patterns, quality, staleness, split suggestions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Full check (all steerings)
  %(prog)s .kiro/steering/code-style.md     # Check one file
  %(prog)s .kiro/steering/                  # Check a folder
  %(prog)s --check quality staleness        # Only specific checks
  %(prog)s --stale-days 30                  # Custom staleness threshold
""",
    )
    parser.add_argument(
        "targets", nargs="*", type=Path, default=None,
        help="Specific steering file(s) or folder to check. Default: all.",
    )
    parser.add_argument(
        "--check", nargs="+", choices=ALL_CHECKS, default=None,
        help=f"Run only these checks. Options: {', '.join(ALL_CHECKS)}",
    )
    parser.add_argument(
        "--stale-days", type=int, default=STALE_DAYS_THRESHOLD,
        help=f"Days threshold for staleness check (default: {STALE_DAYS_THRESHOLD})",
    )
    return parser.parse_args()


def resolve_targets(targets: list[Path] | None) -> list[Path] | None:
    """Resolve CLI targets to a list of steering .md files."""
    if not targets:
        return None  # Use default (all steerings)
    resolved = []
    for t in targets:
        # Resolve relative to PROJECT_ROOT if not absolute
        if not t.is_absolute():
            t = PROJECT_ROOT / t
        if t.is_dir():
            resolved.extend(sorted(t.glob("*.md")))
        elif t.is_file() and t.suffix == ".md":
            resolved.append(t)
        else:
            print(f"  ⚠ Skipping '{t}' (not a .md file or directory)")
    return resolved if resolved else None


def main() -> int:
    global STALE_DAYS_THRESHOLD
    args = parse_args()
    STALE_DAYS_THRESHOLD = args.stale_days
    checks_to_run = set(args.check) if args.check else set(ALL_CHECKS)
    targets = resolve_targets(args.targets)

    # Scan steerings (all or filtered)
    steerings = scan_steerings(targets)

    print("=" * 60)
    print("  Steering Verification Tool")
    print("=" * 60)
    n_scanned = len(targets) if targets else len(list(STEERING_DIR.glob("*.md")))
    print(f"\n  Files: {n_scanned} steerings")
    print(f"  Checks: {', '.join(sorted(checks_to_run))}")
    print(f"  Staleness threshold: {STALE_DAYS_THRESHOLD} days\n")

    all_issues: list[str] = []

    if "activation" in checks_to_run and not targets:
        all_issues.extend(check_activation(steerings))

    if "quality" in checks_to_run:
        all_issues.extend(check_quality(targets))

    if "overlaps" in checks_to_run and not targets:
        all_issues.extend(check_overlaps(steerings))

    if "orphans" in checks_to_run:
        all_issues.extend(check_orphans(steerings))

    if "staleness" in checks_to_run:
        all_issues.extend(check_staleness(steerings))

    if "split" in checks_to_run:
        all_issues.extend(check_split_candidates(targets))

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    if not all_issues:
        print(f"✅ ALL CHECKS PASSED")
        return 0
    else:
        print(f"❌ {len(all_issues)} ISSUES FOUND:")
        for issue in all_issues[:15]:
            print(f"   • {issue}")
        if len(all_issues) > 15:
            print(f"   ... and {len(all_issues) - 15} more")
        return 1


if __name__ == "__main__":
    sys.exit(main())
