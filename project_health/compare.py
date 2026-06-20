# Backward-compatibility shim — module moved to cli/
# ruff: noqa: F401, F403
from project_health.cli.compare import *  # noqa
from project_health.cli.compare import (
    parse_args,
    _run_experiment_comparison,
    _run_noisy_analysis,
    _run_zne_analysis,
    _write_json,
    main,
)
