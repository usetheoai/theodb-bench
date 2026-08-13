"""Explicit representation of a value that was not measured.

A metric that could not be collected must never be serialised as ``0``. Zero is
a legitimate measurement -- zero cache misses, zero failed jobs -- so using it
as a stand-in for "we do not know" silently converts ignorance into evidence.

Every metric in this codebase is therefore ``T | Absent``, and every report
renders an ``Absent`` as its reason rather than as a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias, TypeVar


class AbsenceReason(str, Enum):
    """Why a value is not present. These are not interchangeable."""

    UNSUPPORTED = "unsupported"
    """The system or adapter cannot do this at all (TRD section 27)."""

    UNAVAILABLE = "unavailable"
    """Supported in principle, but not obtainable here -- a perf event the
    kernel does not expose, a statistics view the extension did not install."""

    NOT_COLLECTED = "not_collected"
    """Obtainable, but this run did not ask for it -- collector disabled by
    the profile, or telemetry switched off to reduce overhead."""

    INVALID = "invalid"
    """Collected, but the value failed validation and must not be used."""


@dataclass(frozen=True)
class Absent:
    """A metric slot that holds no measurement, and says why."""

    reason: AbsenceReason
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"absent": self.reason.value}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload

    def __str__(self) -> str:
        return self.reason.value if self.detail is None else f"{self.reason.value}: {self.detail}"


T = TypeVar("T")

Measured: TypeAlias = T | Absent
"""A value that was measured, or an explicit statement that it was not."""


def unsupported(detail: str | None = None) -> Absent:
    return Absent(AbsenceReason.UNSUPPORTED, detail)


def unavailable(detail: str | None = None) -> Absent:
    return Absent(AbsenceReason.UNAVAILABLE, detail)


def not_collected(detail: str | None = None) -> Absent:
    return Absent(AbsenceReason.NOT_COLLECTED, detail)


def invalid(detail: str | None = None) -> Absent:
    return Absent(AbsenceReason.INVALID, detail)


def is_present(value: Measured[T]) -> bool:
    """True when a real measurement is held."""
    return not isinstance(value, Absent)


def encode(value: Measured[Any]) -> Any:
    """Render a measured-or-absent value for JSON serialisation."""
    return value.as_dict() if isinstance(value, Absent) else value
