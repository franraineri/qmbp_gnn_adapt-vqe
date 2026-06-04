"""Integration tests for module API contracts and feasibility checks.

Verifies that all public modules, functions, and classes referenced in
documentation and hardware deployment plans are importable and have the
expected interfaces.

These tests serve as a "contract gate" — if any documented API is broken
or renamed, these tests catch it before runtime failures in experiments.

Usage:
    pytest tests/integration/test_module_contracts.py -v

    # Run only a specific section:
    pytest tests/integration/test_module_contracts.py -k "hardware_config"

    # With custom checks file:
    pytest tests/integration/test_module_contracts.py --contracts-file checks.json
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Fixtures ────────────────────────────────────────────────────────────────


# File existence checks — parameterized for easy extension
REQUIRED_FILES = [
    "src/qmbp_simulation/execution/hardware/backend.py",
    "src/qmbp_simulation/execution/hardware/config.py",
    "src/qmbp_simulation/execution/hardware/submission.py",
    "src/qmbp_simulation/execution/hardware/spsa.py",
    "src/qmbp_simulation/execution/hardware/preflight.py",
    "src/qmbp_simulation/execution/hardware/phase.py",
    "src/qmbp_simulation/execution/hardware/persistence.py",
    "src/qmbp_simulation/execution/hardware/observables.py",
    "src/qmbp_simulation/execution/hardware/README.md",
    "src/qmbp_simulation/execution/backends.py",
    "src/qmbp_simulation/execution/noisy_utils.py",
    "src/qmbp_simulation/framework/runner_base.py",
    "src/qmbp_simulation/framework/criteria.py",
    "scripts/runner_templates/template_validation_runner.py",
]


@pytest.fixture(scope="module")
def project_root() -> Path:
    """Return the project root directory."""
    return ROOT


# ─── Section 1: File existence ───────────────────────────────────────────────


class TestFileExistence:
    """Verify all referenced source files exist on disk."""

    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_file_exists(self, project_root: Path, rel_path: str) -> None:
        full_path = project_root / rel_path
        assert full_path.exists(), f"Required file missing: {rel_path}"


# ─── Section 2: Core imports ─────────────────────────────────────────────────


class TestNoisyUtilsImports:
    """Verify noisy_utils module exports expected symbols."""

    def test_linear_zne_importable(self) -> None:
        from qmbp_simulation.execution.noisy_utils import linear_zne  # noqa: F401

    def test_run_gate_folding_zne_importable(self) -> None:
        from qmbp_simulation.execution.noisy_utils import run_gate_folding_zne  # noqa: F401

    def test_run_pea_zne_importable(self) -> None:
        from qmbp_simulation.execution.noisy_utils import run_pea_zne  # noqa: F401

    def test_noisy_estimator_config_importable(self) -> None:
        from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig  # noqa: F401

    def test_layout_selection_importable(self) -> None:
        from qmbp_simulation.execution.noisy_utils import (  # noqa: F401
            build_adjacency,
            find_layouts_bfs,
            select_layouts_low_ces,
        )

    def test_result_types_importable(self) -> None:
        from qmbp_simulation.execution.noisy_utils import (  # noqa: F401
            GateFoldingZNEResult,
            PEAResult,
        )


class TestMitigationOptionsImports:
    """Verify MitigationOptions has required fields."""

    def test_importable(self) -> None:
        from qmbp_simulation.execution.backends import MitigationOptions  # noqa: F401

    def test_zne_amplifier_field(self) -> None:
        from qmbp_simulation.execution.backends import MitigationOptions

        assert hasattr(MitigationOptions, "zne_amplifier")

    def test_zne_enabled_field(self) -> None:
        from qmbp_simulation.execution.backends import MitigationOptions

        assert hasattr(MitigationOptions, "zne_enabled")

    def test_zne_noise_factors_field(self) -> None:
        from qmbp_simulation.execution.backends import MitigationOptions

        assert hasattr(MitigationOptions, "zne_noise_factors")

    def test_is_dataclass(self) -> None:
        from qmbp_simulation.execution.backends import MitigationOptions

        assert dataclasses.is_dataclass(MitigationOptions)


class TestHardwareConfigImports:
    """Verify HardwareConfig and HardwareRunResult contracts."""

    def test_config_importable(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareConfig  # noqa: F401

    def test_result_importable(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareRunResult  # noqa: F401

    def test_config_mitigation_field(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        field_names = [f.name for f in dataclasses.fields(HardwareConfig)]
        assert "mitigation" in field_names

    def test_config_mode_field(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        assert hasattr(HardwareConfig, "mode")

    def test_result_zne_r2_field(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareRunResult

        field_names = [f.name for f in dataclasses.fields(HardwareRunResult)]
        assert "zne_r2" in field_names

    def test_result_verdict_field(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareRunResult

        assert hasattr(HardwareRunResult, "verdict")

    def test_config_is_dataclass(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        assert dataclasses.is_dataclass(HardwareConfig)

    def test_result_is_dataclass(self) -> None:
        from qmbp_simulation.execution.hardware.config import HardwareRunResult

        assert dataclasses.is_dataclass(HardwareRunResult)


class TestSubmissionImports:
    """Verify hardware submission module contracts."""

    def test_build_estimator_options_importable(self) -> None:
        from qmbp_simulation.execution.hardware.submission import (  # noqa: F401
            build_estimator_options,
        )

    def test_submit_all_then_collect_importable(self) -> None:
        from qmbp_simulation.execution.hardware.submission import (  # noqa: F401
            submit_all_then_collect,
        )

    def test_select_layouts_importable(self) -> None:
        from qmbp_simulation.execution.hardware.submission import (  # noqa: F401
            select_layouts_for_hardware,
        )


class TestHardwareBackendImports:
    """Verify HardwareBackend class contract."""

    def test_importable(self) -> None:
        from qmbp_simulation.execution.hardware.backend import HardwareBackend  # noqa: F401

    def test_run_deployment_method(self) -> None:
        from qmbp_simulation.execution.hardware.backend import HardwareBackend

        assert hasattr(HardwareBackend, "run_deployment")

    def test_evaluate_method(self) -> None:
        from qmbp_simulation.execution.hardware.backend import HardwareBackend

        assert hasattr(HardwareBackend, "evaluate")

    def test_run_preflight_method(self) -> None:
        from qmbp_simulation.execution.hardware.backend import HardwareBackend

        assert hasattr(HardwareBackend, "run_preflight")


class TestRunnerBaseImports:
    """Verify ValidationRunner framework contract."""

    def test_importable(self) -> None:
        from qmbp_simulation.framework.runner_base import ValidationRunner  # noqa: F401

    def test_section_importable(self) -> None:
        from qmbp_simulation.framework.runner_base import Section  # noqa: F401

    def test_resolve_root_importable(self) -> None:
        from qmbp_simulation.framework.runner_base import resolve_project_root  # noqa: F401

    def test_add_custom_args_method(self) -> None:
        from qmbp_simulation.framework.runner_base import ValidationRunner

        assert hasattr(ValidationRunner, "_add_custom_args")

    def test_vqe_descending_sweep_method(self) -> None:
        from qmbp_simulation.framework.runner_base import ValidationRunner

        assert hasattr(ValidationRunner, "vqe_descending_sweep")

    def test_exact_ground_state_method(self) -> None:
        from qmbp_simulation.framework.runner_base import ValidationRunner

        assert hasattr(ValidationRunner, "exact_ground_state")


# ─── Section 3: Framework pattern support ────────────────────────────────────


class TestAmplifierSwitching:
    """Verify build_estimator_options supports ZNE amplifier switching."""

    def _make_config(self, amplifier: str) -> Any:
        from qmbp_simulation.execution.backends import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        return HardwareConfig(
            mode="hardware",
            mitigation=MitigationOptions(zne_enabled=True, zne_amplifier=amplifier),
        )

    def test_gate_folding_config(self) -> None:
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        config = self._make_config("gate_folding")
        opts = build_estimator_options(config)
        resilience = opts.get("resilience", {})
        assert resilience.get("zne_mitigation") is True

    def test_pea_config_amplifier(self) -> None:
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        config = self._make_config("pea")
        opts = build_estimator_options(config)
        resilience = opts.get("resilience", {})
        assert resilience.get("zne", {}).get("amplifier") == "pea"

    def test_pea_config_noise_learning(self) -> None:
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        config = self._make_config("pea")
        opts = build_estimator_options(config)
        resilience = opts.get("resilience", {})
        assert "layer_noise_learning" in resilience


# ─── Section 4: ZNE result type contracts ────────────────────────────────────


class TestZNEResultContracts:
    """Verify ZNE result dataclasses have expected fields."""

    def test_gf_result_extrapolated_value(self) -> None:
        from qmbp_simulation.execution.noisy_utils import GateFoldingZNEResult

        field_names = [f.name for f in dataclasses.fields(GateFoldingZNEResult)]
        assert "extrapolated_value" in field_names

    def test_gf_result_r_squared(self) -> None:
        from qmbp_simulation.execution.noisy_utils import GateFoldingZNEResult

        field_names = [f.name for f in dataclasses.fields(GateFoldingZNEResult)]
        assert "r_squared" in field_names

    def test_pea_result_extrapolated_value(self) -> None:
        from qmbp_simulation.execution.noisy_utils import PEAResult

        field_names = [f.name for f in dataclasses.fields(PEAResult)]
        assert "extrapolated_value" in field_names

    def test_pea_result_r_squared(self) -> None:
        from qmbp_simulation.execution.noisy_utils import PEAResult

        field_names = [f.name for f in dataclasses.fields(PEAResult)]
        assert "r_squared" in field_names


class TestZNEFunctionSignatures:
    """Verify ZNE functions accept expected parameters."""

    def test_gate_folding_signature(self) -> None:
        from qmbp_simulation.execution.noisy_utils import run_gate_folding_zne

        sig = inspect.signature(run_gate_folding_zne)
        assert len(sig.parameters) >= 4, (
            f"run_gate_folding_zne should have ≥4 params, got {len(sig.parameters)}"
        )

    def test_pea_zne_signature(self) -> None:
        from qmbp_simulation.execution.noisy_utils import run_pea_zne

        sig = inspect.signature(run_pea_zne)
        assert len(sig.parameters) >= 4, (
            f"run_pea_zne should have ≥4 params, got {len(sig.parameters)}"
        )


# ─── Section 5: SPSA module contract ────────────────────────────────────────


class TestSPSAContract:
    """Verify SPSA module accepts callable evaluation function."""

    def test_evaluate_fn_parameter(self) -> None:
        from qmbp_simulation.execution.hardware.spsa import spsa_refinement

        sig = inspect.signature(spsa_refinement)
        assert "evaluate_fn" in sig.parameters, (
            "spsa_refinement must accept 'evaluate_fn' parameter"
        )


# ─── Section 6: Criteria module contracts ────────────────────────────────────


class TestCriteriaEntries:
    """Verify experiment criteria registry has expected entries."""

    def test_zne_cross_topo_exists(self) -> None:
        from qmbp_simulation.framework.criteria import EXPERIMENT_CRITERIA

        assert "ZNE_CROSS_TOPO" in EXPERIMENT_CRITERIA

    def test_pea_hw_ready_exists(self) -> None:
        from qmbp_simulation.framework.criteria import EXPERIMENT_CRITERIA

        assert "PEA_HW_READY" in EXPERIMENT_CRITERIA

    def test_gf_zne_cmp_exists(self) -> None:
        from qmbp_simulation.framework.criteria import EXPERIMENT_CRITERIA

        assert "GF_ZNE_CMP" in EXPERIMENT_CRITERIA


# ─── Section 7: ZNE strategy pattern (implementability check) ────────────────


class TestZNEStrategyPattern:
    """Verify the proposed ZNEStrategy pattern is implementable."""

    def test_dataclass_creation(self) -> None:
        """A ZNEStrategy-like dataclass can be constructed and used."""

        @dataclasses.dataclass
        class ZNEStrategy:
            primary_amplifier: str = "gate_folding"
            fallback_amplifier: str = "pea"
            r2_threshold: float = 0.80
            de_gap_threshold: float = 0.10
            max_attempts: int = 2

            def should_retry(self, r2: float, de_gap: float) -> bool:
                return r2 < self.r2_threshold or de_gap > self.de_gap_threshold

        strategy = ZNEStrategy()
        assert strategy.should_retry(0.5, 0.20) is True
        assert strategy.should_retry(0.95, 0.03) is False

    def test_hardware_config_extensible(self) -> None:
        """HardwareConfig being a dataclass means it can be subclassed."""
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        assert dataclasses.is_dataclass(HardwareConfig)
