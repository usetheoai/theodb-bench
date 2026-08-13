"""Benchmark profiles: what a run is allowed to claim.

A profile is not a speed setting. It declares how much rigour was applied and
therefore what the result may be used for. ``smoke`` and ``research`` produce
real measurements that are never publishable evidence, and no amount of
favourable output changes that (PRD section 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from theodb_bench.errors import ConfigError, ErrorContext, Phase


class ProfileName(str, Enum):
    SMOKE = "smoke"
    PR = "pr"
    NIGHTLY = "nightly"
    RELEASE = "release"
    RESEARCH = "research"


class TelemetryLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    EXTENDED = "extended"
    FULL = "full"


@dataclass(frozen=True)
class Profile:
    """The rules a run executed under this profile must satisfy."""

    name: ProfileName
    min_repetitions: int
    telemetry: TelemetryLevel
    isolation_required: bool
    """Whether a failure to apply declared resource controls invalidates the run."""

    preflight_required: bool
    """Whether a failed mandatory preflight check stops the run (TRD 6.0)."""

    publishable: bool
    """Whether a result may back a public performance claim."""

    regression_gate: bool
    """Whether the profile compares against an accepted baseline."""

    frozen_methodology: bool
    """Whether benchmark version, dataset manifests and parameters must be frozen."""

    dirty_tree_invalidates: bool
    """Whether an uncommitted source tree invalidates the run (TRD 6.1)."""

    description: str

    def require_repetitions(self, repetitions: int) -> None:
        """Reject a run configured with fewer repetitions than the profile allows."""
        if repetitions < self.min_repetitions:
            raise ConfigError(
                f"profile {self.name.value!r} requires at least {self.min_repetitions} "
                f"repetition(s); {repetitions} requested",
                context=ErrorContext(phase=Phase.PREFLIGHT, details={"profile": self.name.value}),
            )


PROFILES: Final[dict[ProfileName, Profile]] = {
    ProfileName.SMOKE: Profile(
        name=ProfileName.SMOKE,
        min_repetitions=1,
        telemetry=TelemetryLevel.MINIMAL,
        isolation_required=False,
        preflight_required=False,
        publishable=False,
        regression_gate=False,
        frozen_methodology=False,
        dirty_tree_invalidates=False,
        description="Fast local validation. Never a public performance claim.",
    ),
    ProfileName.PR: Profile(
        name=ProfileName.PR,
        min_repetitions=3,
        telemetry=TelemetryLevel.STANDARD,
        isolation_required=True,
        preflight_required=True,
        publishable=False,
        regression_gate=True,
        frozen_methodology=False,
        dirty_tree_invalidates=False,
        description="Regression detection on controlled benchmark hardware.",
    ),
    ProfileName.NIGHTLY: Profile(
        name=ProfileName.NIGHTLY,
        min_repetitions=3,
        telemetry=TelemetryLevel.EXTENDED,
        isolation_required=True,
        preflight_required=True,
        publishable=False,
        regression_gate=True,
        frozen_methodology=False,
        dirty_tree_invalidates=False,
        description="Larger datasets, more repetitions, broader telemetry.",
    ),
    ProfileName.RELEASE: Profile(
        name=ProfileName.RELEASE,
        min_repetitions=5,
        telemetry=TelemetryLevel.FULL,
        isolation_required=True,
        preflight_required=True,
        publishable=True,
        regression_gate=True,
        frozen_methodology=True,
        dirty_tree_invalidates=True,
        description="Frozen methodology and publishable result bundles.",
    ),
    ProfileName.RESEARCH: Profile(
        name=ProfileName.RESEARCH,
        min_repetitions=1,
        telemetry=TelemetryLevel.STANDARD,
        isolation_required=False,
        preflight_required=False,
        publishable=False,
        regression_gate=False,
        frozen_methodology=False,
        dirty_tree_invalidates=False,
        description="Exploratory work. Results are explicitly non-authoritative.",
    ),
}


def get_profile(name: str) -> Profile:
    """Look up a profile by name, rejecting anything not declared."""
    try:
        key = ProfileName(name)
    except ValueError as exc:
        known = ", ".join(p.value for p in ProfileName)
        raise ConfigError(
            f"unknown profile {name!r}; known profiles: {known}",
            context=ErrorContext(phase=Phase.PREFLIGHT),
            cause=exc,
        ) from exc
    return PROFILES[key]
