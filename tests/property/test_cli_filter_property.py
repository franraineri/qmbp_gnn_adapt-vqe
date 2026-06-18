"""Property-based tests for CLI config filter parsing (resolve_configs).

# Feature: mitigation-benchmark, Property 17
# **Validates: Requirements 2.3**
#
# Property 17: CLI config filter parsing
#   CSV of valid shortnames resolves to correct full config_ids.
#   - For any subset of config shortnames (prefixes that match at least one config),
#     resolve_configs correctly resolves them to the full config_id list.
#   - Order is preserved: resolved configs appear in the order of their shortnames.
#   - Invalid shortnames raise ValueError.
#   - Combined --priority + --configs filtering works correctly (intersection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts to path for benchmark module import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_runners.hardware.benchmark_configs import BENCHMARK_CONFIGS
from experiment_runners.hardware.run_mitigation_benchmark import resolve_configs

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Extract unique prefixes from config keys (the "shortname" portion before "_")
all_config_ids = list(BENCHMARK_CONFIGS.keys())
all_prefixes = sorted(set(cid.split("_")[0] for cid in all_config_ids))

# Strategy for valid prefixes (shortnames that resolve to at least one config)
valid_prefix_st = st.sampled_from(all_prefixes)

# Strategy for subsets of valid prefixes (1-5, unique)
prefix_subset_st = st.lists(valid_prefix_st, min_size=1, max_size=5, unique=True)

# Strategy for random invalid shortnames (strings that don't match any config)
invalid_shortname_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=3,
    max_size=20,
).filter(lambda s: not any(cid.startswith(s) for cid in all_config_ids))

# Strategy for priority level subsets (0-4)
priority_subset_st = st.frozensets(st.integers(min_value=0, max_value=4), min_size=1, max_size=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Property 17: CLI config filter parsing
# **Validates: Requirements 2.3**
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidPrefixesResolveCorrectly:
    """Property 17a: Valid prefixes resolve to all matching config_ids.

    **Validates: Requirements 2.3**

    For any subset of valid config shortnames (prefixes), resolve_configs
    returns a list containing all config_ids that start with each prefix.
    """

    @given(prefixes=prefix_subset_st)
    @settings(max_examples=50, deadline=None)
    def test_valid_prefixes_resolve_to_all_matching_configs(self, prefixes):
        """Each valid prefix resolves to all configs starting with that prefix."""
        args = argparse.Namespace(configs=",".join(prefixes), priority=None)
        result = resolve_configs(args)

        for prefix in prefixes:
            expected = [c for c in all_config_ids if c.startswith(prefix)]
            for exp in expected:
                assert exp in result, (
                    f"Prefix '{prefix}' should resolve to '{exp}', but it's not in result: {result}"
                )

    @given(prefixes=prefix_subset_st)
    @settings(max_examples=50, deadline=None)
    def test_resolved_list_contains_only_prefix_matches(self, prefixes):
        """Resolved list contains ONLY configs that match one of the prefixes."""
        args = argparse.Namespace(configs=",".join(prefixes), priority=None)
        result = resolve_configs(args)

        for config_id in result:
            assert any(config_id.startswith(p) for p in prefixes), (
                f"Config '{config_id}' in result doesn't match any prefix in {prefixes}"
            )


class TestPrefixOrderPreserved:
    """Property 17b: Order is preserved in resolution.

    **Validates: Requirements 2.3**

    Resolved configs appear in the order determined by the shortnames
    in the CSV input. For each shortname, its matching configs appear
    before the next shortname's matching configs.
    """

    @given(prefixes=prefix_subset_st)
    @settings(max_examples=50, deadline=None)
    def test_order_follows_shortname_order(self, prefixes):
        """Resolved configs appear grouped by prefix, in prefix order."""
        args = argparse.Namespace(configs=",".join(prefixes), priority=None)
        result = resolve_configs(args)

        # Build expected ordered list: for each prefix in order,
        # collect all matching config_ids (preserving registry order)
        expected_ordered = []
        for prefix in prefixes:
            matches = [c for c in all_config_ids if c.startswith(prefix)]
            expected_ordered.extend(matches)

        assert result == expected_ordered, (
            f"Order mismatch for prefixes {prefixes}.\n"
            f"Expected: {expected_ordered}\n"
            f"Got:      {result}"
        )


class TestInvalidShortnameFails:
    """Property 17c: Invalid shortnames raise ValueError.

    **Validates: Requirements 2.3**

    Any shortname that doesn't prefix-match any config_id causes
    resolve_configs to raise ValueError.
    """

    @given(invalid=invalid_shortname_st)
    @settings(max_examples=50, deadline=None)
    def test_invalid_shortname_raises_value_error(self, invalid):
        """A shortname that matches no config raises ValueError."""
        args = argparse.Namespace(configs=invalid, priority=None)
        with pytest.raises(ValueError, match="No config matches shortname"):
            resolve_configs(args)


class TestCombinedPriorityAndConfigsFilter:
    """Property 17d: Combined --priority + --configs filtering.

    **Validates: Requirements 2.3**

    When both --priority and --configs are specified, only configs that
    match BOTH the priority filter AND the prefix filter are returned.
    """

    @given(
        prefixes=prefix_subset_st,
        levels=priority_subset_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_combined_filter_is_intersection(self, prefixes, levels):
        """Combined filter returns intersection of priority and prefix matches."""
        # Compute expected: configs matching priority AND prefix
        priority_filtered = [c for c in all_config_ids if BENCHMARK_CONFIGS[c].priority in levels]
        expected = []
        for prefix in prefixes:
            matches = [c for c in priority_filtered if c.startswith(prefix)]
            expected.extend(matches)

        # If any prefix has no match after priority filter, expect ValueError
        has_empty_prefix = any(
            not any(c.startswith(p) for c in priority_filtered) for p in prefixes
        )

        priority_csv = ",".join(f"P{lvl}" for lvl in sorted(levels))
        args = argparse.Namespace(
            configs=",".join(prefixes),
            priority=priority_csv,
        )

        if has_empty_prefix:
            with pytest.raises(ValueError, match="No config matches shortname"):
                resolve_configs(args)
        else:
            result = resolve_configs(args)
            assert result == expected, (
                f"Combined filter mismatch.\n"
                f"Prefixes: {prefixes}, Priority levels: {levels}\n"
                f"Expected: {expected}\n"
                f"Got:      {result}"
            )
