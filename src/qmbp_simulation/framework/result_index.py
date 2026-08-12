"""Result Index — lightweight metadata cache for fast experiment discovery.

Maintains a `.result_index.json` file per experiment directory that caches
key metadata fields from each run_*.json file. This avoids full JSON parsing
when filtering/listing experiments (200+ files → <10ms scan from index).

The index is:
- Auto-updated when `save_experiment_result` writes a new file.
- Auto-rebuilt if the index is stale (run files newer than index).
- Fully backward-compatible: tools that don't use the index still work.

Usage:
    from qmbp_simulation.framework.result_index import ResultIndex

    index = ResultIndex()  # Uses default results/experiments/ root

    # Fast filtered listing (no full JSON parse)
    runs = index.query(model="tfim", topology="heavy_hex", passed=True)

    # Force rebuild (e.g., after manual file edits)
    index.rebuild()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INDEX_FILENAME = ".result_index.json"
_DEFAULT_ROOT = Path("results/experiments")


class ResultIndex:
    """Fast metadata index for experiment results.

    Caches lightweight metadata from each run_*.json to avoid repeated
    full-file parsing during listing/filtering operations.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _DEFAULT_ROOT
        self._entries: list[dict[str, Any]] = []
        self._loaded = False

    def _index_path(self) -> Path:
        return self.root / _INDEX_FILENAME

    def _is_stale(self) -> bool:
        """Check if index needs rebuilding (any run_*.json newer than index)."""
        idx_path = self._index_path()
        if not idx_path.exists():
            return True
        idx_mtime = idx_path.stat().st_mtime
        # Check if any run file is newer than the index
        for f in self.root.rglob("run_*.json"):
            if f.stat().st_mtime > idx_mtime:
                return True
        return False

    def _load_or_rebuild(self) -> None:
        """Load index from file, rebuilding if stale or missing."""
        if self._loaded:
            return

        idx_path = self._index_path()
        if idx_path.exists() and not self._is_stale():
            try:
                with open(idx_path) as f:
                    data = json.load(f)
                self._entries = data.get("runs", [])
                self._loaded = True
                return
            except (json.JSONDecodeError, OSError):
                pass  # Rebuild on corrupt index

        self.rebuild()

    def rebuild(self) -> int:
        """Rebuild the index by scanning all run_*.json files.

        Returns the number of entries indexed.
        """
        from qmbp_simulation.framework.result_io import (
            extract_run_metadata_summary,
            load_result,
        )

        self._entries = []
        if not self.root.exists():
            self._loaded = True
            return 0

        for f in sorted(self.root.rglob("run_*.json")):
            try:
                data = load_result(f)
                summary = extract_run_metadata_summary(data)
                summary["_file"] = str(f.relative_to(self.root))
                self._entries.append(summary)
            except (ValueError, OSError) as e:
                logger.debug("Index skip %s: %s", f.name, e)
                continue

        self._save()
        self._loaded = True
        logger.info("Result index rebuilt: %d entries", len(self._entries))
        return len(self._entries)

    def _save(self) -> None:
        """Persist index to disk."""
        idx_path = self._index_path()
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        index_data = {
            "version": "1.0",
            "n_entries": len(self._entries),
            "runs": self._entries,
        }
        with open(idx_path, "w") as f:
            json.dump(index_data, f, indent=1)

    def add_entry(self, file_path: Path, data: dict[str, Any]) -> None:
        """Add a single entry to the index (called after save_experiment_result).

        Parameters
        ----------
        file_path : Path
            Path to the newly saved run_*.json file.
        data : dict
            The full result envelope that was saved.
        """
        from qmbp_simulation.framework.result_io import extract_run_metadata_summary

        self._load_or_rebuild()
        summary = extract_run_metadata_summary(data)
        try:
            summary["_file"] = str(file_path.relative_to(self.root))
        except ValueError:
            summary["_file"] = str(file_path)
        self._entries.append(summary)
        self._save()

    def query(
        self,
        *,
        model: str | None = None,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
        passed: bool | None = None,
        experiment_id: str | None = None,
        gap_method: str | None = None,
        runner_tag: str | None = None,
        date_tag: str | None = None,
        runner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the index with optional filters.

        All filters are AND-combined. None means "any value".

        Parameters
        ----------
        gap_method : str | None
            Filter by gap computation method. Matches if the specified method
            appears in the run's gap_methods list. Use "eigsh_fallback" to find
            post-fix runs, or "floor_2pi_n" for legacy runs.
        runner_tag : str | None
            Filter by 2-letter runner tag (e.g., "AC" for AcceleratedCrossN,
            "LN" for LargeNExtrapolation). Use for tracing which runner
            produced a result.
        date_tag : str | None
            Filter by date tag in DDMMYY format (e.g., "100826" for Aug 10, 2026).
            Use to find results from a specific day.
        runner_id : str | None
            Filter by runner_id substring match. Use for finding results from
            a specific experiment runner (e.g., "accelerated_cross_n").

        Returns
        -------
        list[dict]
            Matching index entries (lightweight metadata + _file path).
        """
        self._load_or_rebuild()

        results = self._entries
        if model is not None:
            results = [r for r in results if r.get("model") == model]
        if topology is not None:
            results = [r for r in results if r.get("topology") == topology]
        if n_qubits is not None:
            results = [r for r in results if r.get("n_qubits") == n_qubits]
        if p_layers is not None:
            results = [r for r in results if r.get("p_layers") == p_layers]
        if passed is not None:
            results = [r for r in results if r.get("passed") == passed]
        if experiment_id is not None:
            eid_lower = experiment_id.lower()
            results = [r for r in results if eid_lower in r.get("experiment_id", "").lower()]
        if gap_method is not None:
            results = [r for r in results if gap_method in (r.get("gap_methods") or [])]
        if runner_tag is not None:
            results = [r for r in results if r.get("runner_tag") == runner_tag]
        if date_tag is not None:
            results = [r for r in results if r.get("date_tag") == date_tag]
        if runner_id is not None:
            rid_lower = runner_id.lower()
            results = [r for r in results if rid_lower in r.get("runner_id", "").lower()]
        return results

    def get_best_run(
        self,
        *,
        model: str,
        topology: str,
        n_qubits: int | None = None,
        p_layers: int | None = None,
    ) -> dict[str, Any] | None:
        """Find the best (highest pass_rate) run matching the filters.

        Useful for baseline comparison when saving new results.
        """
        candidates = self.query(
            model=model, topology=topology, n_qubits=n_qubits, p_layers=p_layers
        )
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.get("pass_rate", 0), r.get("timestamp", "")))

    @property
    def entries(self) -> list[dict[str, Any]]:
        """All index entries (triggers load/rebuild if needed)."""
        self._load_or_rebuild()
        return self._entries

    def __len__(self) -> int:
        self._load_or_rebuild()
        return len(self._entries)

    # ── Data quality filter ──────────────────────────────────────────────

    @property
    def valid_entries(self) -> list[dict[str, Any]]:
        """Return only entries with sufficient metadata for analysis.

        Excludes entries that are "garbage" — legacy runs from early
        development (TEST, FAIL, NONE, XFAIL experiment_ids), runs
        without proper topology/model/n_qubits, and zero-section runs.

        Criteria for exclusion (ANY triggers exclusion):
        - Empty or invalid topology ('' or '[]')
        - Missing model name
        - Missing or zero n_qubits
        - Missing or zero p_layers
        - Zero sections completed (crashed before any work)
        - Experiment ID is a known test marker (TEST, FAIL, NONE, XFAIL, CNT)
        - pass_rate is not a valid number in [0, 1]

        Returns
        -------
        list[dict]
            Filtered entries suitable for stats, coverage, regressions.
        """
        self._load_or_rebuild()
        _GARBAGE_EXPERIMENT_IDS = {"TEST", "FAIL", "NONE", "XFAIL", "CNT", ""}

        valid = []
        for e in self._entries:
            if not e.get("model"):
                continue
            if e.get("topology") in ("", "[]", None):
                continue
            if not e.get("n_qubits"):
                continue
            if not e.get("p_layers"):
                continue
            if e.get("n_sections", 0) <= 0:
                continue
            if e.get("experiment_id", "").upper() in _GARBAGE_EXPERIMENT_IDS:
                continue
            # Guard: pass_rate must be a valid float in [0, 1]
            pr = e.get("pass_rate")
            if pr is not None:
                try:
                    pr_f = float(pr)
                    if not (0.0 <= pr_f <= 1.0):
                        continue
                except (ValueError, TypeError):
                    continue
            valid.append(e)
        return valid

    # ── A1: Aggregate statistics ─────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Compute aggregate statistics from the index.

        Returns summary with: total_runs, n_passed, n_failed, best per config,
        models/topologies covered, date range.

        Uses only valid entries (excludes garbage/legacy test data).
        """
        valid = self.valid_entries
        if not valid:
            return {"total_runs": 0}

        n_passed = sum(1 for e in valid if e.get("passed"))
        models = sorted(set(e.get("model", "") for e in valid if e.get("model")))
        topologies = sorted(set(e.get("topology", "") for e in valid if e.get("topology")))
        n_values = sorted(set(e.get("n_qubits", 0) for e in valid if e.get("n_qubits")))
        timestamps = [e.get("timestamp", "") for e in valid if e.get("timestamp")]

        return {
            "total_runs": len(valid),
            "n_passed": n_passed,
            "n_failed": len(valid) - n_passed,
            "pass_rate": n_passed / max(len(valid), 1),
            "models": models,
            "topologies": topologies,
            "n_values": n_values,
            "date_range": [min(timestamps), max(timestamps)] if timestamps else [],
            "total_compute_hours": sum(e.get("elapsed_s", 0) for e in valid) / 3600,
        }

    # ── A2: Deduplication ────────────────────────────────────────────────

    def _deduplicate(self) -> int:
        """Remove duplicate entries (same timestamp + experiment_id).

        Returns the number of duplicates removed.
        """
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        n_dupes = 0
        for entry in self._entries:
            key = f"{entry.get('timestamp', '')}|{entry.get('_file', '')}"
            if key in seen:
                n_dupes += 1
                continue
            seen.add(key)
            unique.append(entry)
        self._entries = unique
        if n_dupes > 0:
            self._save()
            logger.info("Deduplicated index: removed %d duplicates", n_dupes)
        return n_dupes

    # ── A3: Index validation ─────────────────────────────────────────────

    def validate(self) -> dict[str, Any]:
        """Validate index integrity: check that referenced files still exist.

        Returns dict with n_valid, n_missing, missing_files.
        """
        self._load_or_rebuild()
        missing: list[str] = []
        for entry in self._entries:
            file_rel = entry.get("_file", "")
            if file_rel:
                full_path = self.root / file_rel
                if not full_path.exists():
                    missing.append(file_rel)
        return {
            "n_valid": len(self._entries) - len(missing),
            "n_missing": len(missing),
            "missing_files": missing[:20],  # Cap at 20 for readability
        }

    # ── B4: Auto-suggest next experiment ─────────────────────────────────

    def suggest_next(self) -> list[str]:
        """Suggest experiments to run based on coverage gaps.

        Identifies (model, topology, N) combinations that either:
        - Have no runs at all
        - Have <80% pass_rate and could benefit from parameter tuning
        """
        self._load_or_rebuild()
        suggestions: list[str] = []

        # Known viable combinations (from documented findings)
        viable_configs = [
            ("tfim", "chain_1d"),
            ("tfim", "heavy_hex"),
            ("tfim_longitudinal", "chain_1d"),
            ("tfim_longitudinal", "heavy_hex"),
        ]
        target_n_values = [10, 16, 20]

        for model, topo in viable_configs:
            for n in target_n_values:
                runs = self.query(model=model, topology=topo, n_qubits=n)
                if not runs:
                    suggestions.append(f"NO DATA: {model} {topo} N={n} — never tested")
                else:
                    best = max(runs, key=lambda r: r.get("pass_rate", 0))
                    rate = best.get("pass_rate", 0)
                    if rate < 0.8:
                        suggestions.append(
                            f"LOW PASS: {model} {topo} N={n} — best={rate:.0%}, "
                            f"try wider h-range or more h-points"
                        )

        return suggestions

    # ── B5: Regression detection ─────────────────────────────────────────

    def detect_regressions(self) -> list[dict[str, Any]]:
        """Find cases where the latest run regressed vs the previous run.

        A regression is: latest run pass_rate < previous run pass_rate
        for the same (model, topology, N, p) config by more than 5%.

        Strategy: compare ONLY the last two runs per config (chronological).
        This avoids false positives from one-off lucky runs in early
        development, and correctly surfaces real degradations.
        """
        valid = self.valid_entries
        from collections import defaultdict

        # Group by config
        groups: dict[str, list[dict]] = defaultdict(list)
        for entry in valid:
            key = (
                f"{entry.get('model', '')}|{entry.get('topology', '')}|"
                f"{entry.get('n_qubits', '')}|{entry.get('p_layers', '')}"
            )
            groups[key].append(entry)

        regressions: list[dict[str, Any]] = []
        for key, runs in groups.items():
            if len(runs) < 2:
                continue
            # Sort by timestamp
            sorted_runs = sorted(runs, key=lambda r: r.get("timestamp", "") or "")
            latest = sorted_runs[-1]
            previous = sorted_runs[-2]

            latest_rate = latest.get("pass_rate", 0)
            prev_rate = previous.get("pass_rate", 0)

            # Regression = latest dropped by more than 5% vs immediate predecessor
            if latest_rate < prev_rate - 0.05:
                regressions.append(
                    {
                        "config": key,
                        "latest_pass_rate": latest_rate,
                        "previous_pass_rate": prev_rate,
                        "best_previous_pass_rate": max(
                            r.get("pass_rate", 0) for r in sorted_runs[:-1]
                        ),
                        "delta": latest_rate - prev_rate,
                        "latest_file": latest.get("_file", ""),
                        "latest_timestamp": latest.get("timestamp", ""),
                        "previous_file": previous.get("_file", ""),
                        "n_previous_runs": len(sorted_runs) - 1,
                    }
                )

        return regressions

    # ── B5b: Temporal regression analysis ────────────────────────────────

    def analyze_temporal_drift(
        self,
        *,
        model: str | None = None,
        topology: str | None = None,
        window_days: int = 7,
    ) -> dict[str, Any]:
        """Analyze whether regressions correlate with time (temporal drift).

        Groups runs by date windows and checks for systematic performance
        degradation over time. Useful for detecting bugs introduced at a
        specific point in time.

        Parameters
        ----------
        model : str | None
            Filter to a specific model (or all if None).
        topology : str | None
            Filter to a specific topology (or all if None).
        window_days : int
            Size of the time window (in days) for bucketing runs (default: 7).

        Returns
        -------
        dict with:
            - "has_drift": bool — whether significant temporal drift detected
            - "windows": list of {date_start, date_end, n_runs, pass_rate}
            - "trend_slope": float — pass_rate change per window (negative = degrading)
            - "breakpoint": str | None — date where performance dropped most
            - "correlation": float — Pearson-like time vs pass_rate correlation
            - "regression_cluster": dict | None — if regressions cluster around a date
        """
        from collections import defaultdict
        from datetime import datetime, timedelta

        entries = self.valid_entries
        if model:
            entries = [e for e in entries if e.get("model") == model]
        if topology:
            entries = [e for e in entries if e.get("topology") == topology]

        if len(entries) < 4:
            return {
                "has_drift": False,
                "windows": [],
                "trend_slope": 0.0,
                "breakpoint": None,
                "correlation": 0.0,
                "regression_cluster": None,
                "n_entries": len(entries),
                "reason": "insufficient data (need >=4 entries)",
            }

        # Parse timestamps safely
        dated_entries: list[tuple[datetime, dict]] = []
        for e in entries:
            ts = e.get("timestamp", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                dated_entries.append((dt, e))
            except (ValueError, TypeError):
                continue

        if len(dated_entries) < 4:
            return {
                "has_drift": False,
                "windows": [],
                "trend_slope": 0.0,
                "breakpoint": None,
                "correlation": 0.0,
                "regression_cluster": None,
                "n_entries": len(dated_entries),
                "reason": "insufficient dated entries (need >=4)",
            }

        dated_entries.sort(key=lambda x: x[0])
        date_min = dated_entries[0][0]
        date_max = dated_entries[-1][0]

        # Bucket into time windows
        window_delta = timedelta(days=window_days)
        windows: list[dict[str, Any]] = []
        current_start = date_min

        while current_start <= date_max:
            current_end = current_start + window_delta
            bucket = [e for dt, e in dated_entries if current_start <= dt < current_end]
            if bucket:
                n_passed = sum(1 for e in bucket if e.get("passed"))
                pass_rate = n_passed / len(bucket)
                windows.append(
                    {
                        "date_start": current_start.strftime("%Y-%m-%d"),
                        "date_end": current_end.strftime("%Y-%m-%d"),
                        "n_runs": len(bucket),
                        "pass_rate": round(pass_rate, 3),
                        "n_passed": n_passed,
                    }
                )
            current_start = current_end

        if len(windows) < 2:
            return {
                "has_drift": False,
                "windows": windows,
                "trend_slope": 0.0,
                "breakpoint": None,
                "correlation": 0.0,
                "regression_cluster": None,
                "n_entries": len(dated_entries),
                "reason": "only one time window",
            }

        # Compute trend: linear regression of pass_rate vs window index
        n_w = len(windows)
        x_vals = list(range(n_w))
        y_vals = [w["pass_rate"] for w in windows]

        x_mean = sum(x_vals) / n_w
        y_mean = sum(y_vals) / n_w

        # Slope via least squares
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals, strict=False))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        slope = numerator / denominator if denominator > 0 else 0.0

        # Pearson correlation
        sy = sum((y - y_mean) ** 2 for y in y_vals)
        correlation = numerator / (denominator * sy) ** 0.5 if denominator > 0 and sy > 0 else 0.0

        # Detect breakpoint: largest single-window drop
        max_drop = 0.0
        breakpoint_date: str | None = None
        for i in range(1, n_w):
            drop = windows[i - 1]["pass_rate"] - windows[i]["pass_rate"]
            if drop > max_drop:
                max_drop = drop
                breakpoint_date = windows[i]["date_start"]

        # Check if regressions cluster temporally
        regressions = self.detect_regressions()
        if model:
            regressions = [r for r in regressions if model in r.get("config", "")]
        if topology:
            regressions = [r for r in regressions if topology in r.get("config", "")]

        regression_cluster: dict[str, Any] | None = None
        if regressions:
            reg_dates: list[str] = []
            for r in regressions:
                ts = r.get("latest_timestamp", "")
                if ts:
                    reg_dates.append(ts[:10])  # date portion only

            if reg_dates:
                # Count regressions per date
                date_counts: dict[str, int] = defaultdict(int)
                for d in reg_dates:
                    date_counts[d] += 1

                peak_date = max(date_counts, key=date_counts.get)  # type: ignore[arg-type]
                if date_counts[peak_date] >= 2:
                    regression_cluster = {
                        "peak_date": peak_date,
                        "n_regressions": date_counts[peak_date],
                        "all_dates": dict(date_counts),
                    }

        # Drift threshold: slope < -0.05 per window (5% drop/week)
        has_drift = slope < -0.05 or max_drop > 0.25

        return {
            "has_drift": has_drift,
            "windows": windows,
            "trend_slope": round(slope, 4),
            "breakpoint": breakpoint_date if max_drop > 0.15 else None,
            "max_single_drop": round(max_drop, 3),
            "correlation": round(correlation, 3),
            "regression_cluster": regression_cluster,
            "n_entries": len(dated_entries),
            "date_range": [
                date_min.strftime("%Y-%m-%d"),
                date_max.strftime("%Y-%m-%d"),
            ],
        }

    # ── B6: Time estimation ──────────────────────────────────────────────

    def estimate_time(
        self, model: str, topology: str, n_qubits: int, p_layers: int
    ) -> float | None:
        """Estimate execution time for a config based on similar previous runs.

        Returns estimated seconds, or None if no similar data available.
        """
        self._load_or_rebuild()

        # Try exact match first
        exact = self.query(model=model, topology=topology, n_qubits=n_qubits, p_layers=p_layers)
        if exact:
            times = [e["elapsed_s"] for e in exact if e.get("elapsed_s", 0) > 0]
            if times:
                return sum(times) / len(times)

        # Fallback: same topology + model, scale by N²
        similar = self.query(model=model, topology=topology)
        if similar:
            # Weight by closeness in N
            weighted_times = []
            for e in similar:
                e_n = e.get("n_qubits", 0)
                e_t = e.get("elapsed_s", 0)
                if e_n > 0 and e_t > 0:
                    # Scale time by (target_N / source_N)^2 (statevector scaling)
                    scale = (n_qubits / e_n) ** 2
                    weighted_times.append(e_t * scale)
            if weighted_times:
                return sum(weighted_times) / len(weighted_times)

        return None

    # ── B7: Coverage matrix ──────────────────────────────────────────────

    def coverage_matrix(self) -> dict[str, dict[str, str]]:
        """Generate a coverage matrix: (model, topology) → latest status.

        Returns nested dict: matrix[model][topology] = "100% (N=20)" or "untested".
        Uses only valid entries (excludes garbage/legacy data).

        Strategy: for each (model, topology) picks the LATEST run per
        (model, topology, N, p) config, then reports the highest pass_rate
        among those latest runs. This avoids inflating the matrix with
        historical peaks that may not be reproducible.
        """
        valid = self.valid_entries
        from collections import defaultdict

        # Step 1: find latest run per full config (model, topo, N, p)
        latest_per_config: dict[tuple, dict] = {}
        for entry in valid:
            model = entry.get("model", "")
            topo = entry.get("topology", "")
            if not model or not topo:
                continue
            key = (model, topo, entry.get("n_qubits", 0), entry.get("p_layers", 0))
            ts = entry.get("timestamp", "") or ""
            prev_ts = latest_per_config.get(key, {}).get("timestamp", "") or ""
            if ts >= prev_ts:
                latest_per_config[key] = entry

        # Step 2: pick best latest-run per (model, topology)
        best: dict[tuple[str, str], dict] = defaultdict(lambda: {"pass_rate": 0, "n": 0})
        for (model, topo, n, _p), entry in latest_per_config.items():
            key = (model, topo)
            rate = entry.get("pass_rate", 0)
            if rate > best[key]["pass_rate"] or (
                rate == best[key]["pass_rate"] and n > best[key]["n"]
            ):
                best[key] = {"pass_rate": rate, "n": n}

        # Build matrix
        models = sorted(set(k[0] for k in best))
        topos = sorted(set(k[1] for k in best))
        matrix: dict[str, dict[str, str]] = {}
        for model in models:
            matrix[model] = {}
            for topo in topos:
                key = (model, topo)
                if key in best:
                    info = best[key]
                    rate = info["pass_rate"]
                    n = info["n"]
                    matrix[model][topo] = f"{rate:.0%} (N={n})"
                else:
                    matrix[model][topo] = "—"
        return matrix

    def print_coverage_matrix(self) -> None:
        """Print the coverage matrix to stdout as a formatted table."""
        matrix = self.coverage_matrix()
        if not matrix:
            print("  No results indexed.")
            return

        models = sorted(matrix.keys())
        topos = sorted(set(t for m in matrix.values() for t in m.keys()))

        # Header
        header = f"{'Model':<25}" + "".join(f"{t:<15}" for t in topos)
        print(header)
        print("-" * len(header))
        for model in models:
            row = f"{model:<25}"
            for topo in topos:
                val = matrix.get(model, {}).get(topo, "—")
                row += f"{val:<15}"
            print(row)

    # ── C1: Diagnose — failure analysis from index metadata ──────────────

    def diagnose(
        self,
        *,
        model: str | None = None,
        topology: str | None = None,
    ) -> dict[str, Any]:
        """Diagnose experiment health by group (model, topology, p_layers).

        Provides per-group failure classification, pass rates, issue detection,
        and actionable recommendations. This consolidates the analysis from
        scan_new_runs.py into the canonical index layer.

        Parameters
        ----------
        model : str | None
            Filter to a specific model (or all if None).
        topology : str | None
            Filter to a specific topology (or all if None).

        Returns
        -------
        dict
            {
                "groups": {config_key: GroupDiagnosis},
                "issues": [str],  # cross-group issues
                "recommendations": [str],
                "summary": {"total_groups", "healthy", "degraded", "failing"},
            }
        """
        from collections import defaultdict

        # Use valid entries only (excludes garbage/legacy data)
        entries = self.valid_entries
        if model:
            entries = [e for e in entries if e.get("model") == model]
        if topology:
            entries = [e for e in entries if e.get("topology") == topology]

        # Group by (model, topology, p_layers)
        groups: dict[str, list[dict]] = defaultdict(list)
        for entry in entries:
            key = (
                f"{entry.get('model', '?')}|"
                f"{entry.get('topology', '?')}|"
                f"p={entry.get('p_layers', '?')}"
            )
            groups[key].append(entry)

        group_diagnoses: dict[str, dict[str, Any]] = {}
        issues: list[str] = []
        recommendations: list[str] = []
        n_healthy = 0
        n_degraded = 0
        n_failing = 0

        for key, runs in sorted(groups.items()):
            n = len(runs)
            n_passed = sum(1 for r in runs if r.get("passed"))
            pass_rate = n_passed / max(n, 1)
            elapsed_avg = sum(r.get("elapsed_s", 0) for r in runs) / max(n, 1)

            # Classify group health
            if pass_rate >= 0.8:
                health = "healthy"
                n_healthy += 1
            elif pass_rate >= 0.4:
                health = "degraded"
                n_degraded += 1
            else:
                health = "failing"
                n_failing += 1

            # Per-group issues
            group_issues: list[str] = []
            if pass_rate < 0.5 and n >= 3:
                group_issues.append(
                    f"Consistently failing ({n_passed}/{n} pass) — "
                    f"check h-range, maxiter, or restart count"
                )
            if n == 1 and not runs[0].get("passed"):
                group_issues.append("Single run, failed — retry with different seed")

            # Detect regression within group (use pass_rate, not binary passed)
            sorted_runs = sorted(runs, key=lambda r: r.get("timestamp", "") or "")
            if len(sorted_runs) >= 2:
                latest_run_rate = sorted_runs[-1].get("pass_rate", 0.0)
                prev_run_rates = [r.get("pass_rate", 0.0) for r in sorted_runs[:-1]]
                prev_avg = sum(prev_run_rates) / len(prev_run_rates)
                if latest_run_rate < prev_avg - 0.2:
                    group_issues.append(
                        f"Latest run degraded vs history "
                        f"({latest_run_rate:.0%} vs avg {prev_avg:.0%})"
                    )

            group_diagnoses[key] = {
                "n_runs": n,
                "n_passed": n_passed,
                "pass_rate": round(pass_rate, 3),
                "health": health,
                "avg_elapsed_s": round(elapsed_avg, 1),
                "issues": group_issues,
                "n_qubits": runs[0].get("n_qubits", 0),
            }

            if group_issues:
                for issue in group_issues:
                    issues.append(f"[{key}] {issue}")

        # Cross-group recommendations
        if n_failing > 0:
            recommendations.append(
                f"{n_failing} config groups are failing — "
                f"prioritize investigation before adding new experiments"
            )
        if n_degraded > n_healthy:
            recommendations.append(
                "More groups degraded than healthy — consider batch re-run with updated params"
            )

        return {
            "groups": group_diagnoses,
            "issues": issues,
            "recommendations": recommendations,
            "summary": {
                "total_groups": len(groups),
                "healthy": n_healthy,
                "degraded": n_degraded,
                "failing": n_failing,
            },
        }

    # ── C2: Refresh project status steering file ─────────────────────────

    def refresh_status(self) -> Path | None:
        """Regenerate .kiro/steering/project-status.md from current index.

        This is the programmatic equivalent of running
        `scripts/update_project_status.py`. Call after saving new results
        to keep Kiro's context up-to-date.

        Returns
        -------
        Path | None
            Path to the updated steering file, or None if generation failed.
        """
        try:
            from datetime import datetime

            stats = self.stats()
            matrix = self.coverage_matrix()
            regressions = self.detect_regressions()
            suggestions = self.suggest_next()

            lines = [
                "# Project Status (Auto-Generated)",
                "",
                f"**Last updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"**Total runs**: {stats.get('total_runs', 0)} | "
                f"Pass: {stats.get('n_passed', 0)} | "
                f"Fail: {stats.get('n_failed', 0)} | "
                f"Rate: {stats.get('pass_rate', 0):.0%}",
                f"**Total compute**: {stats.get('total_compute_hours', 0):.1f} hours",
                f"**Models**: {', '.join(stats.get('models', []))}",
                f"**Topologies**: {', '.join(stats.get('topologies', []))}",
                f"**N values**: {stats.get('n_values', [])}",
                "",
                "## Coverage Matrix (latest pass_rate per config)",
                "",
            ]

            if matrix:
                models = sorted(matrix.keys())
                topos = sorted(set(t for m in matrix.values() for t in m.keys()))
                lines.append("| Model | " + " | ".join(topos) + " |")
                lines.append("|" + "---|" * (len(topos) + 1))
                for m in models:
                    row = f"| {m} |"
                    for t in topos:
                        val = matrix.get(m, {}).get(t, "—")
                        row += f" {val} |"
                    lines.append(row)
                lines.append("")

            if regressions:
                lines.append("## ⚠️ Regressions Detected")
                lines.append("")
                for r in regressions[:5]:
                    lines.append(
                        f"- **{r['config']}**: {r['latest_pass_rate']:.0%} "
                        f"(prev {r['previous_pass_rate']:.0%}, "
                        f"Δ={r['delta']:.0%})"
                    )
                lines.append("")

            if suggestions:
                lines.append("## Suggested Next Experiments")
                lines.append("")
                for s in suggestions[:8]:
                    lines.append(f"- {s}")
                lines.append("")

            # ── Large-N Extrapolation summary ─────────────────────────────
            extrap_section = self._generate_extrapolation_summary()
            if extrap_section:
                lines.extend(extrap_section)

            lines.append("---")
            lines.append("*Generated by `ResultIndex.refresh_status()` from ResultIndex*")

            # Write to steering file
            output_path = Path(".kiro") / "steering" / "project-status.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines))
            logger.debug("Project status refreshed: %s", output_path)
            return output_path

        except Exception as e:
            logger.debug("Could not refresh project status: %s", e)
            return None

    def _generate_extrapolation_summary(self) -> list[str] | None:
        """Generate a compact Large-N Extrapolation summary for project-status.

        Reads NPZ files from data/large_n_extrapolation/ and produces a
        per-topology table with key metrics.
        """
        import numpy as np
        from collections import defaultdict

        extrap_dir = Path("data") / "large_n_extrapolation"
        if not extrap_dir.exists():
            return None

        npz_files = sorted(extrap_dir.glob("*.npz"))
        if not npz_files:
            return None

        topo_data: dict[str, list[dict]] = defaultdict(list)
        for npz_path in npz_files:
            stem = npz_path.stem
            parts = stem.rsplit("_", 2)
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
                per_site = float(abs_errs.mean()) / max(n_qubits, 1)
                if gaps is not None:
                    de_gaps = abs_errs / np.maximum(gaps, 1e-10)
                    mean_dg = float(de_gaps.mean())
                    pass5 = int((de_gaps < 0.05).sum())
                else:
                    mean_dg = -1
                    pass5 = 0
                topo_data[topo].append({
                    "n": n_qubits, "pts": n_pts, "dg": mean_dg,
                    "ps": per_site, "p5": pass5,
                })
            except Exception:
                continue

        if not topo_data:
            return None

        lines = [
            "## Large-N Extrapolation (Zero-Shot MPNN)",
            "",
            "| Topology | N | Pts | ΔE/gap | |ΔE|/N | Pass@5% |",
            "|----------|---|-----|--------|--------|---------|",
        ]
        for topo in sorted(topo_data.keys()):
            entries = sorted(topo_data[topo], key=lambda x: x["n"])
            for e in entries:
                dg_str = f"{e['dg']:.3f}" if e["dg"] >= 0 else "—"
                lines.append(
                    f"| {topo} | {e['n']} | {e['pts']} | "
                    f"{dg_str} | {e['ps']:.2e} | {e['p5']}/{e['pts']} |"
                )
        lines.append("")
        return lines
