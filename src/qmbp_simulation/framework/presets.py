"""Config Presets System — YAML-based experiment configuration.

Provides a way to define experiment configurations as YAML files that can
be loaded by any ValidationRunner via the --preset CLI flag. This replaces
the need to create near-identical runner scripts for each configuration.

Usage:
    # From CLI
    .venv/bin/python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \\
        --preset noiseless/tfim_heavy_hex_n20_p4

    # Programmatic
    from qmbp_simulation.framework.presets import load_preset, list_presets
    preset = load_preset("noiseless/tfim_heavy_hex_n20_p4")
    args_list = preset_to_args(preset)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolve project root (configs/ lives at project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PRESETS_DIR = _PROJECT_ROOT / "configs" / "presets"


def _ensure_yaml() -> Any:
    """Import yaml with a helpful error if not installed."""
    try:
        import yaml

        return yaml
    except ImportError as e:
        raise ImportError("pyyaml is required for presets. Install with: pip install pyyaml") from e


def _load_defaults(category: str) -> dict[str, Any]:
    """Load defaults.yaml for a category if it exists."""
    defaults_path = _PRESETS_DIR / category / "defaults.yaml"
    if not defaults_path.exists():
        return {}
    yaml = _ensure_yaml()
    with open(defaults_path) as f:
        data = yaml.safe_load(f) or {}
    return data


def load_preset(name: str) -> dict[str, Any]:
    """Load a preset by name and merge with category defaults.

    Parameters
    ----------
    name : str
        Preset name in format "category/preset_name" (without .yaml extension).
        E.g. "noiseless/tfim_heavy_hex_n20_p4".

    Returns
    -------
    dict
        Merged preset configuration (defaults + preset-specific overrides).

    Raises
    ------
    FileNotFoundError
        If the preset file doesn't exist.
    ValueError
        If the preset name format is invalid.
    """
    yaml = _ensure_yaml()

    # Support both "category/name" and flat "name" formats
    parts = name.replace("\\", "/").split("/")
    if len(parts) == 1:
        # Search all categories for a matching preset
        preset_path = _find_preset_by_name(parts[0])
    elif len(parts) == 2:
        category, preset_name = parts
        preset_path = _PRESETS_DIR / category / f"{preset_name}.yaml"
    else:
        raise ValueError(f"Invalid preset name '{name}'. Expected 'category/name' or 'name'.")

    if not preset_path.exists():
        available = list_presets()
        raise FileNotFoundError(
            f"Preset '{name}' not found at {preset_path}.\nAvailable presets: {available}"
        )

    # Load category defaults
    category = preset_path.parent.name
    defaults = _load_defaults(category)

    # Load preset
    with open(preset_path) as f:
        preset_data = yaml.safe_load(f) or {}

    # Merge: preset overrides defaults
    merged = {**defaults, **preset_data}
    merged["_preset_name"] = name
    merged["_preset_path"] = str(preset_path)

    logger.info("Loaded preset: %s (%d fields)", name, len(merged) - 2)
    return merged


def _find_preset_by_name(name: str) -> Path:
    """Search all categories for a preset matching the given name."""
    if not _PRESETS_DIR.exists():
        return _PRESETS_DIR / f"{name}.yaml"  # Will raise FileNotFoundError

    for category_dir in sorted(_PRESETS_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        candidate = category_dir / f"{name}.yaml"
        if candidate.exists():
            return candidate

    # Return a path that won't exist (triggers FileNotFoundError in caller)
    return _PRESETS_DIR / f"{name}.yaml"


def list_presets(category: str | None = None) -> list[str]:
    """List available preset names.

    Parameters
    ----------
    category : str | None
        If given, list only presets in that category.
        If None, list all presets across all categories.

    Returns
    -------
    list[str]
        Preset names in "category/name" format (without .yaml extension).
    """
    if not _PRESETS_DIR.exists():
        return []

    presets = []
    dirs_to_scan = [_PRESETS_DIR / category] if category else sorted(_PRESETS_DIR.iterdir())

    for category_dir in dirs_to_scan:
        if not category_dir.is_dir():
            continue
        cat_name = category_dir.name
        for yaml_file in sorted(category_dir.glob("*.yaml")):
            if yaml_file.name == "defaults.yaml":
                continue
            preset_name = yaml_file.stem
            presets.append(f"{cat_name}/{preset_name}")

    return presets


def preset_to_args(preset: dict[str, Any]) -> list[str]:
    """Convert a preset dict to a CLI args list.

    Maps preset YAML keys to their corresponding CLI argument names.
    Only includes keys that map to known physics args.

    Parameters
    ----------
    preset : dict
        Loaded preset configuration.

    Returns
    -------
    list[str]
        CLI-compatible args list, e.g. ["--n-qubits", "20", "--model", "tfim"].
    """
    # Mapping from preset YAML key → CLI flag name
    KEY_TO_FLAG = {
        "n_qubits": "--n-qubits",
        "p_layers": "--p-layers",
        "topology": "--topology",
        "model": "--model",
        "model_params": "--model-params",
        "h_min": "--h-min",
        "h_max": "--h-max",
        "h_points": "--h-points",
        "seeds": "--seeds",
        "maxiter": "--maxiter",
        "n_restarts": "--n-restarts",
        "output": "--output",
    }

    args: list[str] = []
    for key, flag in KEY_TO_FLAG.items():
        if key not in preset:
            continue
        value = preset[key]
        if value is None:
            continue

        # Handle list values (topology, seeds)
        if isinstance(value, list):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.append(flag)
            args.append(str(value))

    return args
