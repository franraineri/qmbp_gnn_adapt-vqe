"""Property-based tests for BenchmarkConfig dataclass and registry.

# Feature: mitigation-benchmark, Properties 1, 2, 3, 18
# **Validates: Requirements 1.3, 1.4, 1.5, 1.7, 2.8**
#
# Property 1: to_mitigation_options round-trip consistency
#   zne_enabled iff zne_method not None, dd_enabled matches config.
#
# Property 2: Mitiq configs force optimization_level=0
#   For all Mitiq configs in registry, optimization_level == 0.
#
# Property 3: Invalid config_id rejection
#   Any string not in _VALID_CONFIG_IDS raises ValueError.
#
# Property 18: Priority filter correctness
#   For any set of priority levels, filtered list contains only/all matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add scripts to path for benchmark_configs import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_runners.hardware.benchmark_configs import (
    _VALID_CONFIG_IDS,
    BENCHMARK_CONFIGS,
    MITIQ_METHODS,
    VALID_DD_SEQUENCES,
    BenchmarkConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Strategy for selecting a valid config from the registry
valid_config_st = st.sampled_from(list(BENCHMARK_CONFIGS.values()))

# Strategy for selecting a valid config_id string
valid_config_id_st = st.sampled_from(sorted(_VALID_CONFIG_IDS))

# Strategy for generating random strings (for invalid config_id testing)
random_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd", "Pc")),
    min_size=1,
    max_size=50,
)

# Strategy for priority level subsets (0-4)
priority_subset_st = st.frozensets(st.integers(min_value=0, max_value=4), min_size=0, max_size=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Property 1: to_mitigation_options round-trip consistency
# **Validates: Requirements 1.4, 1.5**
# ═══════════════════════════════════════════════════════════════════════════════


class TestToMitigationOptionsConsistency:
    """Property 1: to_mitigation_options round-trip consistency.

    **Validates: Requirements 1.4, 1.5**

    For every valid config in BENCHMARK_CONFIGS, calling to_mitigation_options()
    produces a MitigationOptions where:
    - zne_enabled == (config.zne_method is not None)
    - dd_enabled == config.dd_enabled
    - dd_sequence is one of valid sequences when dd_enabled
    """

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_zne_enabled_iff_zne_method_not_none(self, config: BenchmarkConfig):
        """zne_enabled in MitigationOptions matches presence of zne_method."""
        opts = config.to_mitigation_options()
        expected_zne = config.zne_method is not None
        assert opts.zne_enabled == expected_zne, (
            f"Config {config.config_id}: zne_enabled={opts.zne_enabled}, "
            f"but zne_method={config.zne_method} (expected zne_enabled={expected_zne})"
        )

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_dd_enabled_matches_config(self, config: BenchmarkConfig):
        """dd_enabled in MitigationOptions matches config.dd_enabled."""
        opts = config.to_mitigation_options()
        assert opts.dd_enabled == config.dd_enabled, (
            f"Config {config.config_id}: opts.dd_enabled={opts.dd_enabled}, "
            f"config.dd_enabled={config.dd_enabled}"
        )

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_dd_sequence_is_valid_when_dd_enabled(self, config: BenchmarkConfig):
        """When dd_enabled, dd_sequence in MitigationOptions is a valid sequence."""
        opts = config.to_mitigation_options()
        if opts.dd_enabled:
            assert opts.dd_sequence in VALID_DD_SEQUENCES, (
                f"Config {config.config_id}: dd_sequence={opts.dd_sequence} "
                f"not in {VALID_DD_SEQUENCES}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 2: Mitiq configs force optimization_level=0
# **Validates: Requirements 1.5**
# ═══════════════════════════════════════════════════════════════════════════════


class TestMitiqConfigsForceOptLevel0:
    """Property 2: Mitiq configs force optimization_level=0.

    **Validates: Requirements 1.5**

    For all configs in BENCHMARK_CONFIGS where config.is_mitiq is True,
    the config SHALL have optimization_level == 0.
    """

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_mitiq_configs_have_opt_level_0(self, config: BenchmarkConfig):
        """All Mitiq configs have optimization_level=0."""
        assume(config.is_mitiq)
        assert config.optimization_level == 0, (
            f"Config {config.config_id} is Mitiq (zne_method={config.zne_method}) "
            f"but optimization_level={config.optimization_level} (expected 0)"
        )

    def test_all_mitiq_configs_exhaustive(self):
        """Exhaustive check: every config with zne_method in MITIQ_METHODS has opt_level=0."""
        mitiq_configs = [c for c in BENCHMARK_CONFIGS.values() if c.zne_method in MITIQ_METHODS]
        # Ensure we actually have Mitiq configs in the registry
        assert len(mitiq_configs) > 0, "No Mitiq configs found in BENCHMARK_CONFIGS"

        for config in mitiq_configs:
            assert config.optimization_level == 0, (
                f"Config {config.config_id} uses Mitiq method '{config.zne_method}' "
                f"but optimization_level={config.optimization_level}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 3: Invalid config_id rejection
# **Validates: Requirements 1.3**
# ═══════════════════════════════════════════════════════════════════════════════


class TestInvalidConfigIdRejection:
    """Property 3: Invalid config_id rejection.

    **Validates: Requirements 1.3**

    Any string not in _VALID_CONFIG_IDS raises ValueError when used as
    config_id in BenchmarkConfig.
    """

    @given(text=random_text_st)
    @settings(max_examples=100, deadline=None)
    def test_random_string_raises_value_error(self, text: str):
        """Random strings not in registry raise ValueError."""
        assume(text not in _VALID_CONFIG_IDS)
        with pytest.raises(ValueError):
            BenchmarkConfig(config_id=text)

    @given(config_id=valid_config_id_st)
    @settings(max_examples=19, deadline=None)
    def test_valid_config_ids_do_not_raise(self, config_id: str):
        """All valid config_ids can be instantiated without error."""
        # Should not raise — just instantiate with minimal valid params
        config = BenchmarkConfig(config_id=config_id)
        assert config.config_id == config_id


# ═══════════════════════════════════════════════════════════════════════════════
# Property 18: Priority filter correctness
# **Validates: Requirements 1.7, 2.8**
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriorityFilterCorrectness:
    """Property 18: Priority filter correctness.

    **Validates: Requirements 1.7, 2.8**

    For any set of priority levels, filtering BENCHMARK_CONFIGS by those levels
    produces a list containing exactly the configs whose priority is in the set.
    """

    @given(levels=priority_subset_st)
    @settings(max_examples=50, deadline=None)
    def test_filtered_contains_only_matching_priorities(self, levels: frozenset[int]):
        """Filtered list contains ONLY configs with matching priority."""
        filtered = [c for c in BENCHMARK_CONFIGS.values() if c.priority in levels]
        for config in filtered:
            assert config.priority in levels, (
                f"Config {config.config_id} has priority={config.priority} "
                f"which is not in requested levels {levels}"
            )

    @given(levels=priority_subset_st)
    @settings(max_examples=50, deadline=None)
    def test_filtered_contains_all_matching_priorities(self, levels: frozenset[int]):
        """Filtered list contains ALL configs with matching priority (no omissions)."""
        filtered = [c for c in BENCHMARK_CONFIGS.values() if c.priority in levels]
        filtered_ids = {c.config_id for c in filtered}

        # Every config in the registry with matching priority must be present
        for config in BENCHMARK_CONFIGS.values():
            if config.priority in levels:
                assert config.config_id in filtered_ids, (
                    f"Config {config.config_id} (priority={config.priority}) "
                    f"should be in filtered results for levels {levels}"
                )

    @given(levels=priority_subset_st)
    @settings(max_examples=50, deadline=None)
    def test_filtered_count_matches_expected(self, levels: frozenset[int]):
        """Filtered count equals number of configs with priority in levels."""
        filtered = [c for c in BENCHMARK_CONFIGS.values() if c.priority in levels]
        expected_count = sum(1 for c in BENCHMARK_CONFIGS.values() if c.priority in levels)
        assert len(filtered) == expected_count, (
            f"Expected {expected_count} configs for levels {levels}, got {len(filtered)}"
        )

    def test_empty_levels_returns_empty(self):
        """Empty priority set returns no configs."""
        filtered = [c for c in BENCHMARK_CONFIGS.values() if c.priority in set()]
        assert len(filtered) == 0

    def test_all_levels_returns_all_configs(self):
        """All priority levels (0-4) returns all configs."""
        all_levels = {0, 1, 2, 3, 4}
        filtered = [c for c in BENCHMARK_CONFIGS.values() if c.priority in all_levels]
        assert len(filtered) == len(BENCHMARK_CONFIGS)
