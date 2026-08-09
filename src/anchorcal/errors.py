"""Typed failures used to stop the pilot at scientific gates."""


class AnchorCalError(RuntimeError):
    """Base class for an expected, user-actionable AnchorCal failure."""


class ConfigurationError(AnchorCalError):
    """The resolved configuration violates a locked decision."""


class PreflightError(AnchorCalError):
    """A dataset, mask, model, environment, or provenance precondition failed."""


class AuditFailure(AnchorCalError):
    """A hard leakage, competence, numerical, or diversity gate failed."""


class StorageError(AnchorCalError):
    """Transactional artifact storage is inconsistent or incomplete."""

