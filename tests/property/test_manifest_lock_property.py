"""Property-based test for manifest file-lock integrity under concurrent writes.

# Feature: mitigation-benchmark, Property 8
# **Validates: Requirements 5.4**
#
# Property 8: Manifest file-lock integrity
#   Concurrent writes to manifest.json produce valid JSON without data loss.
#   After N threads each append a unique entry, the resulting manifest contains
#   exactly N valid entries with no corruption or missing data.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts to path for benchmark module import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from experiment_runners.hardware.run_mitigation_benchmark import append_to_manifest

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Strategy for number of concurrent writers
n_writers_st = st.integers(min_value=2, max_value=10)

# Strategy for h_value (realistic range)
h_value_st = st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)

# Strategy for delta_e_gap (realistic range)
delta_e_gap_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for correct_label
correct_label_st = st.booleans()

# Expected manifest entry keys
MANIFEST_ENTRY_KEYS = {
    "config_id",
    "execution_mode",
    "h_value",
    "timestamp",
    "result_path",
    "delta_e_gap",
    "correct_label",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Property 8: Manifest file-lock integrity
# **Validates: Requirements 5.4**
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestFileLockIntegrity:
    """Property 8: Manifest file-lock integrity.

    **Validates: Requirements 5.4**

    Concurrent writes to manifest.json produce valid JSON without data loss.
    After N threads each append a unique entry, the resulting manifest is:
    - Valid JSON (parseable without error)
    - A JSON array of exactly N entries
    - Every entry is a dict with expected keys
    - All N unique entries are present (no data loss)
    """

    @given(n_writers=n_writers_st)
    @settings(max_examples=5, deadline=30000)
    def test_concurrent_manifest_writes_preserve_all_entries(self, n_writers: int):
        """Concurrent writes produce valid JSON with all entries present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"

            # Build unique entries for each writer
            entries = [
                {
                    "config_id": f"C{i}_test",
                    "execution_mode": "fake_backend",
                    "h_value": 3.25 + i * 0.1,
                    "timestamp": f"2026-06-18T10:{i:02d}:00Z",
                    "result_path": f"fake_backend/C{i}_test/h3p25_run.json",
                    "delta_e_gap": 0.01 * (i + 1),
                    "correct_label": i % 2 == 0,
                }
                for i in range(n_writers)
            ]

            # Spawn concurrent writers
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers) as executor:
                futures = [
                    executor.submit(append_to_manifest, entry, manifest_path) for entry in entries
                ]
                concurrent.futures.wait(futures)

            # Check no exceptions were raised
            for future in futures:
                future.result()  # Raises if the thread raised

            # --- Verify valid JSON ---
            raw_content = manifest_path.read_text()
            data = json.loads(raw_content)  # Must not raise

            # --- Verify is a JSON array ---
            assert isinstance(data, list), (
                f"Manifest must be a JSON array, got {type(data).__name__}"
            )

            # --- Verify length matches n_writers ---
            assert len(data) == n_writers, f"Expected {n_writers} entries, got {len(data)}"

            # --- Verify all entries are dicts with expected keys ---
            for idx, item in enumerate(data):
                assert isinstance(item, dict), f"Entry {idx} is not a dict: {type(item).__name__}"
                assert MANIFEST_ENTRY_KEYS.issubset(item.keys()), (
                    f"Entry {idx} missing keys: {MANIFEST_ENTRY_KEYS - set(item.keys())}"
                )

            # --- Verify no data loss: all unique config_ids present ---
            written_config_ids = {entry["config_id"] for entry in entries}
            manifest_config_ids = {item["config_id"] for item in data}
            assert written_config_ids == manifest_config_ids, (
                f"Data loss detected. Missing: {written_config_ids - manifest_config_ids}"
            )

    @given(
        n_writers=n_writers_st,
        h_value=h_value_st,
        delta_e_gap=delta_e_gap_st,
        correct_label=correct_label_st,
    )
    @settings(max_examples=5, deadline=30000)
    def test_concurrent_writes_preserve_structure(
        self,
        n_writers: int,
        h_value: float,
        delta_e_gap: float,
        correct_label: bool,
    ):
        """After concurrent writes, manifest entries have correct structure.

        Each entry must be a dict with keys: config_id, execution_mode,
        h_value, timestamp, result_path, delta_e_gap, correct_label.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"

            entries = [
                {
                    "config_id": f"C{i}_struct_test",
                    "execution_mode": "fake_backend",
                    "h_value": h_value,
                    "timestamp": f"2026-06-18T10:{i:02d}:00Z",
                    "result_path": f"fake_backend/C{i}_struct/result.json",
                    "delta_e_gap": delta_e_gap,
                    "correct_label": correct_label,
                }
                for i in range(n_writers)
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers) as executor:
                futures = [
                    executor.submit(append_to_manifest, entry, manifest_path) for entry in entries
                ]
                concurrent.futures.wait(futures)

            for future in futures:
                future.result()

            data = json.loads(manifest_path.read_text())

            # Every entry must have correct types
            for item in data:
                assert isinstance(item["config_id"], str)
                assert isinstance(item["execution_mode"], str)
                assert isinstance(item["h_value"], (int, float))
                assert isinstance(item["timestamp"], str)
                assert isinstance(item["result_path"], str)
                assert isinstance(item["delta_e_gap"], (int, float))
                assert isinstance(item["correct_label"], bool)

    @given(n_writers=n_writers_st)
    @settings(max_examples=5, deadline=30000)
    def test_manifest_created_from_scratch(self, n_writers: int):
        """Manifest is created correctly when it does not exist before writes.

        Verifies append_to_manifest creates parent directories and the
        manifest file when they don't exist, even under concurrent access.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use a nested subdirectory to test parent creation
            manifest_path = Path(tmp_dir) / "nested" / "manifest.json"

            assert not manifest_path.parent.exists()

            entries = [
                {
                    "config_id": f"C{i}_new",
                    "execution_mode": "hardware",
                    "h_value": 3.5,
                    "timestamp": f"2026-06-18T11:{i:02d}:00Z",
                    "result_path": f"hardware/C{i}_new/result.json",
                    "delta_e_gap": 0.02,
                    "correct_label": True,
                }
                for i in range(n_writers)
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers) as executor:
                futures = [
                    executor.submit(append_to_manifest, entry, manifest_path) for entry in entries
                ]
                concurrent.futures.wait(futures)

            for future in futures:
                future.result()

            data = json.loads(manifest_path.read_text())
            assert len(data) == n_writers
            assert all(item["config_id"].endswith("_new") for item in data)
