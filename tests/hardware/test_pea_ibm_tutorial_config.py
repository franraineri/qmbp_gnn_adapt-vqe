"""Tests for PEA IBM tutorial configuration integration (2026-06-17).

Validates that the new PEA configuration fields (layer_pair_depths,
twirling_strategy, extrapolator, ibm_canonical preset) are correctly
integrated across the full pipeline:
  MitigationOptions → HardwareConfig → build_estimator_options →
  _apply_estimator_options → persistence snapshot.

Reference: IBM PEA tutorial (2026), Kim et al. Nature 618 (2023).
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "scripts/experiment_runners/hardware")


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 1: MitigationOptions new fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestMitigationOptionsNewFields:
    """Verify new fields exist with correct defaults."""

    def test_layer_pair_depths_default_none(self):
        from qmbp_simulation.execution import MitigationOptions

        m = MitigationOptions()
        assert m.layer_pair_depths is None

    def test_layer_pair_depths_explicit(self):
        from qmbp_simulation.execution import MitigationOptions

        m = MitigationOptions(layer_pair_depths=[0, 1, 2, 4, 8])
        assert m.layer_pair_depths == [0, 1, 2, 4, 8]

    def test_twirling_strategy_default_none(self):
        from qmbp_simulation.execution import MitigationOptions

        m = MitigationOptions()
        assert m.twirling_strategy is None

    def test_twirling_strategy_explicit(self):
        from qmbp_simulation.execution import MitigationOptions

        m = MitigationOptions(twirling_strategy="active-circuit")
        assert m.twirling_strategy == "active-circuit"

    def test_ibm_tutorial_depths(self):
        """IBM tutorial uses [0,1,2,4,6,12,24] for deep Trotter circuits."""
        from qmbp_simulation.execution import MitigationOptions

        ibm_deep = MitigationOptions(layer_pair_depths=[0, 1, 2, 4, 6, 12, 24])
        assert len(ibm_deep.layer_pair_depths) == 7
        assert ibm_deep.layer_pair_depths[0] == 0
        assert ibm_deep.layer_pair_depths[-1] == 24

    def test_hva_p1_depths(self):
        """HVA p=1 (1 layer of 2Q gates) needs fewer depths."""
        from qmbp_simulation.execution import MitigationOptions

        hva = MitigationOptions(layer_pair_depths=[0, 1, 2, 4, 8])
        assert len(hva.layer_pair_depths) == 5
        assert max(hva.layer_pair_depths) == 8

    def test_backward_compatible(self):
        """Existing code creating MitigationOptions without new fields still works."""
        from qmbp_simulation.execution import MitigationOptions

        # This is how existing tests create it — must not break
        m = MitigationOptions(
            zne_enabled=True,
            zne_amplifier="pea",
            num_randomizations=32,
            shots_per_randomization=128,
        )
        assert m.layer_pair_depths is None
        assert m.twirling_strategy is None
        assert m.zne_amplifier == "pea"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 2: HardwareConfig defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardwareConfigDefaults:
    """Verify HardwareConfig integrates new fields correctly."""

    def test_default_twirling_strategy_active_circuit(self):
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        c = HardwareConfig()
        assert c.mitigation.twirling_strategy == "active-circuit"

    def test_default_layer_pair_depths_none(self):
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        c = HardwareConfig()
        assert c.mitigation.layer_pair_depths is None

    def test_default_pea_amplifier(self):
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        c = HardwareConfig()
        assert c.mitigation.zne_amplifier == "pea"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 3: build_estimator_options integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildEstimatorOptions:
    """Verify build_estimator_options correctly propagates new fields."""

    def test_layer_pair_depths_not_sent_when_none(self):
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig()
        opts = build_estimator_options(c)
        lnl = opts["resilience"]["layer_noise_learning"]
        assert "layer_pair_depths" not in lnl

    def test_layer_pair_depths_sent_when_explicit(self):
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="pea",
                num_randomizations=40,
                shots_per_randomization=64,
                layer_pair_depths=[0, 1, 2, 4, 8],
            )
        )
        opts = build_estimator_options(c)
        lnl = opts["resilience"]["layer_noise_learning"]
        assert lnl["layer_pair_depths"] == [0, 1, 2, 4, 8]

    def test_twirling_strategy_sent_when_set(self):
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="pea",
                twirling_strategy="active-circuit",
            )
        )
        opts = build_estimator_options(c)
        assert opts["twirling"]["strategy"] == "active-circuit"

    def test_twirling_strategy_not_sent_when_none(self):
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="gate_folding",
                twirling_strategy=None,
            )
        )
        opts = build_estimator_options(c)
        assert "strategy" not in opts["twirling"]

    def test_extrapolator_sent_for_pea(self):
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig()  # default is PEA
        opts = build_estimator_options(c)
        zne = opts["resilience"]["zne"]
        assert zne["extrapolator"] == ("exponential", "linear")

    def test_extrapolator_not_sent_for_gate_folding(self):
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                zne_enabled=True,
                zne_amplifier="gate_folding",
                zne_noise_factors=[1, 3, 5],
            )
        )
        opts = build_estimator_options(c)
        zne = opts["resilience"]["zne"]
        assert "extrapolator" not in zne

    def test_gate_folding_no_layer_noise_learning(self):
        """Gate-folding should NOT send layer_noise_learning options."""
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="gate_folding",
            )
        )
        opts = build_estimator_options(c)
        assert "layer_noise_learning" not in opts.get("resilience", {})

    def test_fake_backend_mode_unchanged(self):
        """fake_backend mode should return minimal config (no mitigation)."""
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(mode="fake_backend")
        opts = build_estimator_options(c)
        assert "default_precision" in opts
        assert "resilience" not in opts
        assert "twirling" not in opts


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 4: PEA Presets (ibm_canonical)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPEAPresets:
    """Verify PEA presets including the new ibm_canonical."""

    def test_ibm_canonical_exists(self):
        from run_ibm_deployment import PEA_PRESETS

        assert "ibm_canonical" in PEA_PRESETS

    def test_ibm_canonical_noise_factors(self):
        """IBM PEA tutorial uses [1, 1.3, 1.6] — gentle factors for PEA."""
        from run_ibm_deployment import PEA_PRESETS

        num_r, shots_r, nf, n_lay = PEA_PRESETS["ibm_canonical"]
        assert nf == [1, 1.3, 1.6]

    def test_ibm_canonical_learning_budget(self):
        """IBM tutorial uses 40 rand × 64 shots = 2560 learning shots."""
        from run_ibm_deployment import PEA_PRESETS

        num_r, shots_r, nf, n_lay = PEA_PRESETS["ibm_canonical"]
        assert num_r == 40
        assert shots_r == 64
        assert num_r * shots_r == 2560

    def test_ibm_canonical_layouts(self):
        from run_ibm_deployment import PEA_PRESETS

        _, _, _, n_lay = PEA_PRESETS["ibm_canonical"]
        assert n_lay == 3

    def test_all_presets_have_4_elements(self):
        from run_ibm_deployment import PEA_PRESETS

        for name, preset in PEA_PRESETS.items():
            assert len(preset) == 4, f"Preset '{name}' should have 4 elements, got {len(preset)}"

    def test_build_hardware_config_ibm_canonical(self):
        from run_ibm_deployment import build_hardware_config

        cfg = build_hardware_config(pea_preset="ibm_canonical")
        assert cfg.mitigation.zne_noise_factors == [1, 1.3, 1.6]
        assert cfg.mitigation.num_randomizations == 40
        assert cfg.mitigation.shots_per_randomization == 64
        assert cfg.mitigation.twirling_strategy == "active-circuit"
        assert cfg.n_layouts == 3

    def test_build_hardware_config_with_layer_pair_depths(self):
        from run_ibm_deployment import build_hardware_config

        cfg = build_hardware_config(
            pea_preset="balanced",
            layer_pair_depths=[0, 1, 2, 4, 6, 12],
        )
        assert cfg.mitigation.layer_pair_depths == [0, 1, 2, 4, 6, 12]

    def test_build_hardware_config_layer_pair_depths_default_none(self):
        from run_ibm_deployment import build_hardware_config

        cfg = build_hardware_config(pea_preset="balanced")
        assert cfg.mitigation.layer_pair_depths is None

    def test_build_hardware_config_invalid_preset_raises(self):
        from run_ibm_deployment import build_hardware_config

        with pytest.raises(ValueError, match="Unknown PEA preset"):
            build_hardware_config(pea_preset="nonexistent")


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 5: Persistence snapshot
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistenceSnapshot:
    """Verify persistence snapshot captures new fields."""

    def test_snapshot_includes_layer_pair_depths(self):
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.persistence import (
            _build_mitigation_snapshot,
        )

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="pea",
                layer_pair_depths=[0, 1, 2, 4, 8],
                twirling_strategy="active-circuit",
            )
        )
        snap = _build_mitigation_snapshot(c)
        assert snap["techniques"]["pea_noise_learning"]["layer_pair_depths"] == [0, 1, 2, 4, 8]

    def test_snapshot_includes_twirling_strategy(self):
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.persistence import (
            _build_mitigation_snapshot,
        )

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="pea",
                twirling_strategy="active-circuit",
            )
        )
        snap = _build_mitigation_snapshot(c)
        assert snap["techniques"]["pauli_twirling"]["strategy"] == "active-circuit"

    def test_snapshot_layer_pair_depths_none(self):
        """When layer_pair_depths is None, snapshot should record None."""
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.persistence import (
            _build_mitigation_snapshot,
        )

        c = HardwareConfig()
        snap = _build_mitigation_snapshot(c)
        assert snap["techniques"]["pea_noise_learning"]["layer_pair_depths"] is None

    def test_snapshot_twirling_strategy_from_default_config(self):
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.persistence import (
            _build_mitigation_snapshot,
        )

        c = HardwareConfig()
        snap = _build_mitigation_snapshot(c)
        assert snap["techniques"]["pauli_twirling"]["strategy"] == "active-circuit"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 6: End-to-end pipeline integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndIntegration:
    """Verify the full flow: preset → config → options_dict → apply."""

    def test_ibm_canonical_full_flow(self):
        """ibm_canonical preset produces correct options dict end-to-end."""
        from run_ibm_deployment import build_hardware_config

        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        cfg = build_hardware_config(pea_preset="ibm_canonical")
        opts = build_estimator_options(cfg)

        # ZNE options
        zne = opts["resilience"]["zne"]
        assert zne["amplifier"] == "pea"
        assert zne["noise_factors"] == [1, 1.3, 1.6]
        assert zne["extrapolator"] == ("exponential", "linear")

        # Layer noise learning
        lnl = opts["resilience"]["layer_noise_learning"]
        assert lnl["num_randomizations"] == 40
        assert lnl["shots_per_randomization"] == 64
        assert "layer_pair_depths" not in lnl  # None → not sent

        # Twirling
        tw = opts["twirling"]
        assert tw["strategy"] == "active-circuit"
        assert tw["enable_gates"] is True
        assert tw["num_randomizations"] == 40
        assert tw["shots_per_randomization"] == 64

        # DD
        dd = opts["dynamical_decoupling"]
        assert dd["enable"] is True
        assert dd["sequence_type"] == "XpXm"

        # TREX
        assert opts["resilience"]["measure_mitigation"] is True

    def test_balanced_with_depths_full_flow(self):
        """Balanced + explicit depths produces correct options."""
        from run_ibm_deployment import build_hardware_config

        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        cfg = build_hardware_config(
            pea_preset="balanced",
            layer_pair_depths=[0, 1, 2, 4, 8],
        )
        opts = build_estimator_options(cfg)

        lnl = opts["resilience"]["layer_noise_learning"]
        assert lnl["layer_pair_depths"] == [0, 1, 2, 4, 8]
        assert lnl["num_randomizations"] == 48
        assert lnl["shots_per_randomization"] == 192

    def test_aggressive_full_flow(self):
        """Aggressive preset: maximum budget."""
        from run_ibm_deployment import build_hardware_config

        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        cfg = build_hardware_config(pea_preset="aggressive")
        opts = build_estimator_options(cfg)

        zne = opts["resilience"]["zne"]
        assert zne["noise_factors"] == [1, 1.5, 2, 3]
        assert zne["extrapolator"] == ("exponential", "linear")

        lnl = opts["resilience"]["layer_noise_learning"]
        assert lnl["num_randomizations"] == 64
        assert lnl["shots_per_randomization"] == 256

    def test_gate_folding_excludes_pea_fields(self):
        """Gate-folding mode must NOT send PEA-specific fields."""
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="gate_folding",
                zne_noise_factors=[1, 3, 5],
            )
        )
        opts = build_estimator_options(c)

        # No layer_noise_learning for gate_folding
        assert "layer_noise_learning" not in opts["resilience"]
        # No extrapolator for gate_folding
        assert "extrapolator" not in opts["resilience"]["zne"]
        # No twirling strategy (None default)
        assert "strategy" not in opts.get("twirling", {})
        # No num_randomizations in twirling for gate_folding
        assert "num_randomizations" not in opts.get("twirling", {})


# ═══════════════════════════════════════════════════════════════════════════════
# Test Group 7: Bug fixes
# ═══════════════════════════════════════════════════════════════════════════════


class TestBugFixes:
    """Verify bug fixes discovered during implementation."""

    def test_adaptive_maps_to_pea_for_runtime(self):
        """Bug fix: 'adaptive' is local-only. Runtime should receive 'pea'.

        IBM Runtime only accepts: "pea", "gate_folding", "gate_folding_front",
        "gate_folding_back". Sending "adaptive" would cause a server-side error.
        """
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="adaptive",
                zne_noise_factors=[1, 3, 5],
            )
        )
        opts = build_estimator_options(c)
        zne = opts["resilience"]["zne"]
        # Must send "pea" to Runtime, NOT "adaptive"
        assert zne["amplifier"] == "pea", (
            f"Expected 'pea' for Runtime (adaptive is local-only), got '{zne['amplifier']}'"
        )

    def test_adaptive_gets_layer_noise_learning(self):
        """adaptive → pea on Runtime, so layer_noise_learning should be sent."""
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="adaptive",
                num_randomizations=48,
                shots_per_randomization=192,
            )
        )
        opts = build_estimator_options(c)
        assert "layer_noise_learning" in opts["resilience"]
        lnl = opts["resilience"]["layer_noise_learning"]
        assert lnl["num_randomizations"] == 48
        assert lnl["shots_per_randomization"] == 192

    def test_adaptive_gets_extrapolator(self):
        """adaptive → pea on Runtime, so extrapolator should be sent."""
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="adaptive",
            )
        )
        opts = build_estimator_options(c)
        zne = opts["resilience"]["zne"]
        assert zne["extrapolator"] == ("exponential", "linear")

    def test_adaptive_gets_twirling_randomizations(self):
        """adaptive → pea on Runtime, so twirling budget should be set."""
        from qmbp_simulation.execution import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.submission import build_estimator_options

        c = HardwareConfig(
            mitigation=MitigationOptions(
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier="adaptive",
                num_randomizations=48,
                shots_per_randomization=192,
            )
        )
        opts = build_estimator_options(c)
        tw = opts["twirling"]
        assert tw["num_randomizations"] == 48
        assert tw["shots_per_randomization"] == 192
