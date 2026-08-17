"""Vulture whitelist — symbols used dynamically or by Python conventions.

These are NOT dead code; vulture reports them because it can't see their usage:
- Signal handler args: (signum, frame) — required by signal.signal() API
- Context manager args: (exc_type, exc_val, exc_tb) — required by __exit__() protocol
- Variables unpacked but intentionally unused (e.g., resumed_data)

Note: vulture treats any assignment in this file as a "whitelist usage".
The variable names must match exactly what vulture reports.
"""

# Signal handler signature: def handler(signum, frame)
signum  # type: ignore
frame  # type: ignore

# Context manager __exit__(self, exc_type, exc_val, exc_tb)
exc_val  # type: ignore
exc_tb  # type: ignore

# Variables that are computed for later use or checkpoint recovery
resumed_data  # type: ignore
resumed_sections  # type: ignore

# Configuration flag used conditionally
require_variational  # type: ignore
