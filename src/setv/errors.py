"""Project-specific exceptions."""


class SETVError(RuntimeError):
    """Base error for an invalid SETV artifact or operation."""


class ConfigurationError(SETVError):
    """Raised when configuration is missing or internally inconsistent."""


class DataValidationError(SETVError):
    """Raised when data do not satisfy a locked preflight requirement."""


class ArtifactExistsError(SETVError):
    """Raised when an immutable artifact destination already exists."""

