"""Typed errors carrying the context needed to diagnose a failed run.

Every error raised by this package carries where it happened (phase, system,
benchmark, run) and whether the caller may retry. A benchmark that swallows an
error reports a number that was never measured, so errors here are loud and
specific by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    """Run lifecycle phases (TRD section 6)."""

    PREFLIGHT = "preflight"
    ENVIRONMENT = "environment"
    ISOLATION = "isolation"
    BOOTSTRAP = "bootstrap"
    DATASET_LOAD = "dataset_load"
    INDEX_BUILD = "index_build"
    WARMUP = "warmup"
    MEASUREMENT = "measurement"
    COOLDOWN = "cooldown"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    # Outside the measured lifecycle: dataset management, re-analysis, reporting.
    OFFLINE = "offline"


class Recoverability(str, Enum):
    """Whether retrying the same operation could plausibly succeed.

    A network timeout is recoverable; a violated business rule is not. Treating
    the two the same is how a benchmark ends up retrying its way into a wrong
    number (TRD section 6).
    """

    RECOVERABLE = "recoverable"
    FATAL = "fatal"


@dataclass
class ErrorContext:
    """Where an error happened. All fields optional: an error raised before a
    run exists still carries whatever context was known."""

    phase: Phase | None = None
    system: str | None = None
    benchmark: str | None = None
    run_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.phase is not None:
            out["phase"] = self.phase.value
        if self.system is not None:
            out["system"] = self.system
        if self.benchmark is not None:
            out["benchmark"] = self.benchmark
        if self.run_id is not None:
            out["run_id"] = self.run_id
        if self.details:
            out["details"] = dict(self.details)
        return out


class BenchError(Exception):
    """Base class for every error this package raises."""

    recoverability: Recoverability = Recoverability.FATAL

    def __init__(
        self,
        message: str,
        *,
        context: ErrorContext | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context if context is not None else ErrorContext()
        self.cause = cause

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": type(self).__name__,
            "message": self.message,
            "recoverability": self.recoverability.value,
        }
        payload.update(self.context.as_dict())
        if self.cause is not None:
            payload["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return payload

    def __str__(self) -> str:
        ctx = self.context.as_dict()
        ctx.pop("details", None)
        if not ctx:
            return self.message
        rendered = " ".join(f"{k}={v}" for k, v in sorted(ctx.items()))
        return f"{self.message} [{rendered}]"


class ConfigError(BenchError):
    """Malformed or contradictory user-supplied configuration."""


class SchemaValidationError(BenchError):
    """A machine-readable artifact does not conform to its declared schema."""


class PreflightError(BenchError):
    """A mandatory preflight check failed; the run must not start (TRD 6.0)."""


class DatasetError(BenchError):
    """Dataset acquisition, verification, or preprocessing failed."""


class ChecksumMismatchError(DatasetError):
    """A dataset file's checksum does not match its manifest.

    Never recoverable by retrying: the bytes on disk are not the bytes the
    manifest identifies, so any measurement over them describes a different
    dataset than the one the result would claim.
    """


class AdapterError(BenchError):
    """A system adapter failed to perform a lifecycle operation."""


class SystemUnavailableError(AdapterError):
    """The system under test could not be reached or did not become ready."""

    recoverability = Recoverability.RECOVERABLE


class UnsupportedCapabilityError(AdapterError):
    """The adapter does not support the requested workload feature.

    This is not a failure of the run: it yields an explicit `unsupported`
    result rather than a fabricated measurement (TRD section 27).
    """


class IsolationError(BenchError):
    """A declared resource control could not be applied or was violated."""


class ProcessEscapedError(IsolationError):
    """A process escaped the declared resource controls (TRD 10.2)."""


class MeasurementError(BenchError):
    """The measurement window did not produce usable data."""


class RunValidationError(BenchError):
    """The run completed but failed protocol validation (TRD 6.9)."""


class ImmutableBundleError(BenchError):
    """An attempt was made to modify a finalized run bundle (TRD 3, D3)."""


class ComparabilityError(BenchError):
    """Two results were compared that the protocol does not allow comparing.

    Regression comparison fails closed rather than silently comparing across
    incompatible hardware or configuration (TRD section 22).
    """
