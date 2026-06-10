#!/usr/bin/env python
"""Audit Affine Overshoot Frequency — Close coverage gap G8.

Scans all ZNE experiment results to count how often ZNE extrapolation
produces energies below the exact ground state (overshoot). Documents
the frequency and magnitude of cases where affine_correct_energy() would
intervene.

This answers: "How often does ZNE overshoot, and by how much?"

Output:
    results/gnn_qem/affine_overshoot_audit.json
    (also prints summary to stdout)

Usage:
    python scripts/audit_affine_overshoot.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Experiment dirs that contain ZNE results
ZNE_EXPERIMENT_DIRS = [
    "results/experiments/exp_gf_zne_cmp",
    "results/experiments/exp_zne_3way",
    "results/experiments/exp_pea_zne_val",
    "results/experiments/exp_pea_hw_ready",
    "results/experiments/exp_pea_pipeline",
    "results/experiments/exp_zne_cross_topo",
    "results/experiments/exp_pea_triangular",
    "results/experiments/exp_noisy_variants",
]


def _extract_from_row(row: dict, filepath: Path) -> list[dict]:
    """Extract overshoot records from a single result row."""
    records = []
    e_exact = row.get("e_exact")
    if e_exact is None or not np.isfinite(e_exact):
        return records

    # All known ZNE energy field names across experiment formats
    zne_energy_fields = [
        "e_ces_zne",
        "e_gf_zne",
        "e_pea_zne",
        "e_zne",
        "e_pea",
        "e_gf",
        "e_noisy_raw",
        "extrapolated_value",
        "pea_energy",
        "gf_energy",
    ]
    for key in zne_energy_fields:
        if key in row:
            e_zne = row[key]
            if e_zne is not None and np.isfinite(e_zne):
                records.append(
                    {
                        "file": str(filepath.name),
                        "e_exact": e_exact,
                        "e_zne": e_zne,
                        "overshoot": e_zne < e_exact,
                        "overshoot_magnitude": max(0.0, e_exact - e_zne),
                        "method": key,
                    }
                )
    return records


def scan_experiment_file(filepath: Path) -> list[dict]:
    """Extract per-h-point ZNE results from a single experiment JSON."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    records = []
    results = data.get("results", {})

    # Handle list-based results
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict):
                records.extend(_extract_from_row(row, filepath))
        return records

    # Handle section-based format (most ZNE experiments)
    if isinstance(results, dict):
        # First pass: build e_exact lookup by h from any section that has it
        e_exact_by_h: dict[float, float] = {}
        for section_key, section_data in results.items():
            if not isinstance(section_data, dict):
                continue
            inner = section_data.get("data", section_data)
            if isinstance(inner, dict):
                rows = inner.get("results", inner.get("comparison", []))
            elif isinstance(inner, list):
                rows = inner
            else:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and "e_exact" in row and "h" in row:
                    e_exact_by_h[row["h"]] = row["e_exact"]

        # Second pass: extract ZNE energies, inject e_exact from lookup
        for section_key, section_data in results.items():
            if not isinstance(section_data, dict):
                continue
            inner = section_data.get("data", section_data)
            if isinstance(inner, dict):
                rows = inner.get("results", inner.get("comparison", []))
            elif isinstance(inner, list):
                rows = inner
            else:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # If row doesn't have e_exact, try to inject from lookup
                if "e_exact" not in row and "h" in row:
                    h_val = row["h"]
                    if h_val in e_exact_by_h:
                        row = {**row, "e_exact": e_exact_by_h[h_val]}
                records.extend(_extract_from_row(row, filepath))

    # Also check flat results (exp_noisy_variants format)
    for key in ["noisy_results", "zne_results", "per_h_results"]:
        if key in data and isinstance(data[key], list):
            for item in data[key]:
                if isinstance(item, dict):
                    records.extend(_extract_from_row(item, filepath))

    return records


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "results" / "gnn_qem"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    files_scanned = 0

    for dir_rel in ZNE_EXPERIMENT_DIRS:
        dir_path = project_root / dir_rel
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue
        for json_file in sorted(dir_path.glob("*.json")):
            records = scan_experiment_file(json_file)
            all_records.extend(records)
            files_scanned += 1

    if not all_records:
        logger.warning("No ZNE energy records found. Checking raw field presence...")
        # Fallback: scan for any file with extrapolated values
        for dir_rel in ZNE_EXPERIMENT_DIRS:
            dir_path = project_root / dir_rel
            if not dir_path.exists():
                continue
            for json_file in sorted(dir_path.glob("*.json")):
                try:
                    with open(json_file) as f:
                        content = f.read()
                    if "extrapolated" in content or "e_zne" in content:
                        logger.info(f"  Has ZNE data: {json_file.name}")
                except OSError:
                    pass
        logger.info("Run with raw data inspection to extract overshoot cases.")
        # Still produce output with zero records
    else:
        logger.info(f"Scanned {files_scanned} files, found {len(all_records)} ZNE energy records")

    # Analysis
    n_total = len(all_records)
    n_overshoot = sum(1 for r in all_records if r["overshoot"])
    overshoot_rate = n_overshoot / max(n_total, 1) * 100

    magnitudes = [r["overshoot_magnitude"] for r in all_records if r["overshoot"]]
    mean_magnitude = float(np.mean(magnitudes)) if magnitudes else 0.0
    max_magnitude = float(np.max(magnitudes)) if magnitudes else 0.0

    # Per-method breakdown
    methods_seen = set(r["method"] for r in all_records)
    per_method = {}
    for method in sorted(methods_seen):
        method_records = [r for r in all_records if r["method"] == method]
        n_m = len(method_records)
        n_overshoot_m = sum(1 for r in method_records if r["overshoot"])
        per_method[method] = {
            "n_records": n_m,
            "n_overshoot": n_overshoot_m,
            "overshoot_rate_pct": n_overshoot_m / max(n_m, 1) * 100,
        }

    summary = {
        "n_files_scanned": files_scanned,
        "n_zne_records": n_total,
        "n_overshoot": n_overshoot,
        "overshoot_rate_pct": overshoot_rate,
        "mean_overshoot_magnitude": mean_magnitude,
        "max_overshoot_magnitude": max_magnitude,
        "affine_correction_would_help": n_overshoot,
        "per_method": per_method,
        "conclusion": (
            f"Affine correction intervenes in {overshoot_rate:.1f}% of ZNE extrapolations. "
            f"Mean correction magnitude: {mean_magnitude:.4f} energy units."
            if n_total > 0
            else "No ZNE energy records found for audit."
        ),
    }

    logger.info(f"\n{'=' * 60}")
    logger.info("AFFINE OVERSHOOT AUDIT")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Files scanned: {files_scanned}")
    logger.info(f"  ZNE records: {n_total}")
    logger.info(f"  Overshoot cases: {n_overshoot} ({overshoot_rate:.1f}%)")
    if magnitudes:
        logger.info(f"  Mean overshoot: {mean_magnitude:.4f}")
        logger.info(f"  Max overshoot:  {max_magnitude:.4f}")
    logger.info(f"  Conclusion: affine_correct_energy() helps {overshoot_rate:.1f}% of the time")
    logger.info(f"{'=' * 60}")

    # Save
    audit_path = output_dir / "affine_overshoot_audit.json"
    with open(audit_path, "w") as f:
        json.dump({"summary": summary, "records": all_records[:100]}, f, indent=2, default=str)
    logger.info(f"Saved to {audit_path}")


if __name__ == "__main__":
    main()
