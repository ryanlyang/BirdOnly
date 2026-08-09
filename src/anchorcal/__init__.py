"""AnchorCal Waterbirds100 pilot.

The package is intentionally independent of :mod:`setv`.  Importing it never
loads torch, touches the network, or resolves machine-local paths.
"""

from .errors import AnchorCalError, AuditFailure, ConfigurationError, PreflightError

__all__ = [
    "AnchorCalError",
    "AuditFailure",
    "ConfigurationError",
    "PreflightError",
]

__version__ = "0.6.0"
