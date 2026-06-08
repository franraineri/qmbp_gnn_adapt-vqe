"""Shared CLI argument parsing for scripts and experiments.

Provides reusable argument groups and validation functions that eliminate
boilerplate across scripts/run_pipeline.py, scripts/benchmark.py,
scripts/run_experiment.py, and other entry points.

Usage:
    from qmbp_simulation.framework.cli import (
        create_base_parser,
        add_system_args,
        add_sweep_args,
        add_vqe_args,
        add_mpnn_args,
        add_output_args,
        validate_descending_sweep,
    )

    parser = create_base_parser("My Script", epilog="Examples: ...")
    add_system_args(parser)
    add_sweep_args(parser)
    add_output_args(parser)
    args = parser.parse_args()
    h_values = validate_descending_sweep(args.h_values)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np


def create_base_parser(
    description: str,
    epilog: str = "",
) -> argparse.ArgumentParser:
    """Create a standardized ArgumentParser with consistent formatting.

    Parameters
    ----------
    description : str
        Script description shown in --help.
    epilog : str
        Examples section shown after arguments in --help.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with RawDescriptionHelpFormatter.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )


def add_system_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard system configuration arguments.

    Adds: --n-qubits, --topology, --J, --periodic, --p

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group (for further customization if needed).
    """
    group = parser.add_argument_group("System configuration")
    group.add_argument(
        "--n-qubits",
        type=int,
        default=6,
        help="Number of qubits (default: 6)",
    )
    group.add_argument(
        "--topology",
        type=str,
        default="chain_1d",
        help="Lattice topology (default: chain_1d)",
    )
    group.add_argument(
        "--J",
        type=float,
        default=1.0,
        help="Coupling constant (default: 1.0)",
    )
    group.add_argument(
        "--periodic",
        action="store_true",
        help="Use periodic boundary conditions",
    )
    group.add_argument(
        "--p",
        type=int,
        default=2,
        help="HVA layers (default: 2, max: 2)",
    )
    return group


def add_sweep_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard sweep configuration arguments.

    Adds: --h-values, --h-test

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("Sweep configuration")
    group.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        help="Transverse field values (descending). Default: linspace(2.0, 0.5, 31)",
    )
    group.add_argument(
        "--h-test",
        nargs="+",
        type=float,
        default=[1.5],
        help="Unseen h-value(s) for deployment (default: 1.5)",
    )
    return group


def add_vqe_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard VQE configuration arguments.

    Adds: --n-restarts, --maxiter, --sigma

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("VQE configuration")
    group.add_argument(
        "--n-restarts",
        type=int,
        default=5,
        help="VQE restarts (default: 5)",
    )
    group.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="VQE max iterations (default: 1000)",
    )
    group.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Initial parameter spread for restarts (default: 0.1)",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for VQE + MPNN reproducibility (default: None → non-deterministic)",
    )
    return group


def add_mpnn_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard MPNN configuration arguments.

    Adds: --hidden-dim, --n-layers, --n-epochs, --lr, --patience

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("MPNN configuration")
    group.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="MPNN hidden dimension (default: 128)",
    )
    group.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="MPNN message-passing layers (default: 3)",
    )
    group.add_argument(
        "--n-epochs",
        type=int,
        default=6000,
        help="MPNN training epochs (default: 6000)",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="MPNN learning rate (default: 1e-3)",
    )
    group.add_argument(
        "--patience",
        type=int,
        default=500,
        help="MPNN early stopping patience (default: 500)",
    )
    return group


def add_output_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard output and logging arguments.

    Adds: --output-dir, --verbose, --debug

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("Output")
    group.add_argument(
        "--output-dir",
        type=str,
        default="results/pipeline",
        help="Output directory (default: results/pipeline)",
    )
    group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO logging",
    )
    group.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return group


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────


def validate_descending_sweep(
    h_values: list[float] | None,
    default_start: float = 2.0,
    default_end: float = 0.5,
    default_n: int = 31,
) -> np.ndarray:
    """Validate and normalize h-values to descending order.

    If h_values is None, generates a default linspace.
    If provided, sorts into descending order.

    Parameters
    ----------
    h_values : list[float] | None
        User-provided h-values, or None for default.
    default_start : float
        Start of default linspace (highest h).
    default_end : float
        End of default linspace (lowest h).
    default_n : int
        Number of points in default linspace.

    Returns
    -------
    np.ndarray
        h-values in descending order.

    Raises
    ------
    ValueError
        If h_values is empty after filtering.
    """
    if h_values is None:
        return np.linspace(default_start, default_end, default_n)

    arr = np.array(sorted(h_values, reverse=True), dtype=float)
    if len(arr) == 0:
        raise ValueError("h_values cannot be empty.")
    return arr


def validate_system_size(n_qubits: int, p_layers: int) -> list[str]:
    """Validate system size against known constraints.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    p_layers : int
        Number of HVA layers.

    Returns
    -------
    list[str]
        List of warnings (empty if all OK).

    Raises
    ------
    ValueError
        If p_layers > 2 (hard constraint).
    """
    if p_layers > 2:
        raise ValueError(
            "CONSTRAINT VIOLATION: p_layers > 2 is forbidden (Mele et al. 2022). "
            "HVA depth is limited to p ≤ 2."
        )

    warnings: list[str] = []
    if n_qubits == 12:
        warnings.append("N=12 is very slow (>30 min per run). Consider N=10 or N=14.")
    if n_qubits > 20:
        warnings.append(f"N={n_qubits} may be infeasible for full VQE.")
    return warnings


def configure_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging level based on CLI flags.

    Parameters
    ----------
    verbose : bool
        Enable INFO level logging.
    debug : bool
        Enable DEBUG level logging (overrides verbose).
    """
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_mpnn_config_dict(args: argparse.Namespace) -> dict:
    """Extract MPNN config from parsed args into a dict for PipelineRunner.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (must have hidden_dim, n_layers, n_epochs, lr, patience).

    Returns
    -------
    dict
        MPNN configuration dictionary.
    """
    config = {}
    if hasattr(args, "hidden_dim") and args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim
    if hasattr(args, "n_layers") and args.n_layers is not None:
        config["n_layers"] = args.n_layers
    if hasattr(args, "n_epochs") and args.n_epochs is not None:
        config["n_epochs"] = args.n_epochs
    if hasattr(args, "lr") and args.lr is not None:
        config["lr"] = args.lr
    if hasattr(args, "patience") and args.patience is not None:
        config["patience"] = args.patience
    return config


def resolve_output_dir(path: str | Path) -> Path:
    """Resolve and create output directory.

    Parameters
    ----------
    path : str | Path
        Output directory path.

    Returns
    -------
    Path
        Resolved Path object (directory is created if needed).
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Validation arguments
# ─────────────────────────────────────────────────────────────────────────────


def add_validation_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add VQE and θ validation arguments.

    Adds: --validate-vqe, --validate-theta, --theta-validation-level,
          --strict-validation

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("Validation")
    group.add_argument(
        "--validate-vqe",
        action="store_true",
        default=True,
        help=(
            "Run VQE result validation (variational principle, energy bounds, "
            "convergence checks). Default: on."
        ),
    )
    group.add_argument(
        "--no-validate-vqe",
        action="store_false",
        dest="validate_vqe",
        help="Disable VQE result validation.",
    )
    group.add_argument(
        "--validate-theta",
        action="store_true",
        default=True,
        help=("Run θ_pred validation after MPNN inference (levels 1-4). Default: on."),
    )
    group.add_argument(
        "--no-validate-theta",
        action="store_false",
        dest="validate_theta",
        help="Disable θ_pred validation.",
    )
    group.add_argument(
        "--theta-validation-level",
        type=int,
        default=4,
        choices=range(1, 8),
        metavar="[1-7]",
        help=(
            "Maximum θ validation level (1=bounds, 2=NaN, 3=interpolation, "
            "4=fidelity, 5=gradient, 6=MC-dropout, 7=sensitivity). "
            "Levels 5-7 have significant computational cost. Default: 4."
        ),
    )
    group.add_argument(
        "--strict-validation",
        action="store_true",
        default=False,
        help=(
            "Abort execution on CRITICAL validation failures instead of "
            "logging and continuing. Default: off."
        ),
    )
    return group


# ─────────────────────────────────────────────────────────────────────────────
# Noisy Simulation / ZNE arguments
# ─────────────────────────────────────────────────────────────────────────────


def add_noisy_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard noisy simulation and ZNE configuration arguments.

    Adds: --zne-amplifier, --zne-noise-factors, --zne-extrapolator,
          --zne-shots, --zne-n-layouts, --zne-multi-layout

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("ZNE / Noisy Simulation")
    group.add_argument(
        "--zne-amplifier",
        choices=["gate_folding", "pea", "adaptive"],
        default="gate_folding",
        help=(
            "ZNE noise amplification strategy (default: gate_folding). "
            "'gate_folding': digital U→U·U†·U (simple, validated). "
            "'pea': Probabilistic Error Amplification (learns noise model, "
            "more accurate but ~50%% overhead). "
            "'adaptive': try gate_folding first, fall back to PEA if R²<threshold."
        ),
    )
    group.add_argument(
        "--zne-noise-factors",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Noise amplification factors (default: [1, 3, 5]). "
            "Gate-folding requires odd integers; PEA allows any float ≥1."
        ),
    )
    group.add_argument(
        "--zne-extrapolator",
        choices=["linear", "exponential"],
        default="linear",
        help="Extrapolation method for ZNE (default: linear)",
    )
    group.add_argument(
        "--zne-r2-threshold",
        type=float,
        default=0.90,
        help=(
            "R² threshold for adaptive ZNE fallback (default: 0.90). "
            "Only used when --zne-amplifier=adaptive."
        ),
    )
    group.add_argument(
        "--zne-shots",
        type=int,
        default=16384,
        help="Shots for noisy ZNE estimation (default: %(default)s)",
    )
    group.add_argument(
        "--zne-n-layouts",
        type=int,
        default=3,
        help="Number of low-CES layouts for ZNE (default: %(default)s)",
    )
    group.add_argument(
        "--zne-multi-layout",
        action="store_true",
        help="Run ZNE on ALL layouts and average (variance reduction)",
    )
    return group


# ─────────────────────────────────────────────────────────────────────────────
# Result Filtering and Output Format argument groups
# ─────────────────────────────────────────────────────────────────────────────


def add_result_filter_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard result filtering arguments.

    Adds: --topology, --n-qubits, --p-layers, --model, --folder

    These filters are shared across scripts/digest, scripts/compare, and
    any tool that queries experiment results.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("Result filters")
    group.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Filter by topology (chain_1d, ladder, triangular, kagome, heavy_hex)",
    )
    group.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Filter by system size",
    )
    group.add_argument(
        "--p-layers",
        type=int,
        default=None,
        help="Filter by HVA depth (1 or 2)",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model type (tfim, tfim_longitudinal, heisenberg)",
    )
    group.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specific results folder to scan",
    )
    return group


def add_format_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard output format arguments.

    Adds: --markdown, --json, --output, --sort, --top, --group-by

    These formatting options are shared across digest, compare, and
    other result-reporting scripts.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("Output format")
    group.add_argument(
        "--markdown",
        action="store_true",
        help="Output in Markdown format",
    )
    group.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="FILE",
        help="Save output as JSON to file",
    )
    group.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save text/markdown output to file",
    )
    group.add_argument(
        "--sort",
        type=str,
        default=None,
        help="Sort key (varies by context: delta_e, time, r2, gain, id, verdict)",
    )
    group.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N results (after sorting)",
    )
    group.add_argument(
        "--group-by",
        type=str,
        default=None,
        dest="group_by",
        help="Group results by dimension (topology, n_qubits, p_layers, etc.)",
    )
    return group


def add_variant_runner_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add standard variant runner arguments.

    Adds: --dry-run, --variant, --start-from, --list

    These arguments are shared across all topology-specific variant runner
    scripts (chain_1d, ladder, triangular, kagome, heavy_hex).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.

    Returns
    -------
    argparse._ArgumentGroup
        The argument group.
    """
    group = parser.add_argument_group("Variant runner")
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing",
    )
    group.add_argument(
        "--variant",
        type=int,
        default=None,
        metavar="IDX",
        help="Run only this variant index (0-based)",
    )
    group.add_argument(
        "--start-from",
        type=int,
        default=0,
        metavar="IDX",
        help="Start from this variant index (default: 0)",
    )
    group.add_argument(
        "--list",
        action="store_true",
        dest="list_variants",
        help="List all variants without executing",
    )
    return group
