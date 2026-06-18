"""Property-based test for error resilience of the benchmark runner.

Feature: mitigation-benchmark, Property 15
**Validates: Requirements 2.6**

Property 15: Error resilience — runner continues after failure
  Exceptions raised by run_single_config() do not abort the run_benchmark
  loop. All configs × h_values are attempted regardless of prior failures,
  and _save_error_result is called for each failing execution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts to path for benchmark_configs import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_runners.hardware import run_mitigation_benchmark as runner_mod
from experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Non-AQC configs (use provided h_values, not config.h_test_values)
_NON_AQC_CONFIGS = [cid for cid, cfg in BENCHMARK_CONFIGS.items() if not cfg.aqc_enabled]

# Subset of configs for test (2-5 non-AQC configs to keep h_values predictable)
configs_subset_st = st.lists(
    st.sampled_from(_NON_AQC_CONFIGS),
    min_size=2,
    max_size=5,
    unique=True,
)

# Which configs should fail (random subset count)
n_failing_st = st.integers(min_value=1, max_value=3)

# h-values for test (single value keeps assertions straightforward)
h_values_st = st.just([3.5])


# ═══════════════════════════════════════════════════════════════════════════════
# Property 15: Error resilience — runner continues after failure
# ═══════════════════════════════════════════════════════════════════════════════


@given(
    configs=configs_subset_st,
    n_failing=n_failing_st,
    h_values=h_values_st,
)
@settings(max_examples=15, deadline=30000)
def test_runner_continues_after_failures(
    configs: list[str],
    n_failing: int,
    h_values: list[float],
) -> None:
    """**Validates: Requirements 2.6**

    Property: run_benchmark() calls run_single_config for ALL configs × h_values
    even when some raise exceptions. The total number of attempts equals
    len(configs) × len(h_values) — no config is skipped due to a prior failure.
    Additionally, _save_error_result is called exactly once per failure.
    """
    # Clamp n_failing to not exceed config count
    n_failing = min(n_failing, len(configs))
    failing_configs = set(configs[:n_failing])

    # Track all calls manually
    call_log: list[str] = []
    error_log: list[str] = []

    original_run_single = runner_mod.run_single_config

    def mock_run_single(config, h_value, mode, shots, seed):
        """Track call and raise RuntimeError for configs in failing set."""
        call_log.append(config.config_id)
        if config.config_id in failing_configs:
            raise RuntimeError(f"Simulated failure for {config.config_id}")
        return {"results": {"delta_e_gap": 0.05}}

    def mock_save_error(config, h_value, mode, seed, error):
        """Track error persistence calls."""
        error_log.append(config.config_id)

    with (
        patch.object(runner_mod, "run_single_config", side_effect=mock_run_single),
        patch.object(runner_mod, "_save_error_result", side_effect=mock_save_error),
    ):
        # Execute — should NOT raise despite internal failures
        runner_mod.run_benchmark(configs, h_values, "fake_backend", 16384, 42)

    # ── Assertions ───────────────────────────────────────────────────────
    expected_total = len(configs) * len(h_values)

    # 1. ALL configs were attempted (no early abort)
    assert len(call_log) == expected_total, (
        f"Expected {expected_total} calls to run_single_config, "
        f"got {len(call_log)}. Runner aborted early."
    )

    # 2. Every config_id appears exactly len(h_values) times in call log
    for config_id in configs:
        count = call_log.count(config_id)
        assert count == len(h_values), (
            f"Config {config_id} called {count} times, expected {len(h_values)}"
        )

    # 3. _save_error_result called exactly for failing configs × h_values
    expected_errors = n_failing * len(h_values)
    assert len(error_log) == expected_errors, (
        f"Expected {expected_errors} error saves, got {len(error_log)}."
    )

    # 4. Only failing configs appear in error log
    for config_id in error_log:
        assert config_id in failing_configs, (
            f"Non-failing config {config_id} had _save_error_result called."
        )
