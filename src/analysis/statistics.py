"""Aggregating measurements without hiding them.

Three rules from the protocol shape this module.

Every repetition is retained. Aggregates carry the values they came from, so a
reader can see the spread rather than take the median on trust.

No outlier is removed silently. The only policy that applies by default is
``none``; anything else must be named, versioned and recorded in the artifact.

Throughput is best-of-N, following the ANN-Benchmarks protocol: the reciprocal
of the fastest per-round mean. The dispersion reported alongside it is
within-sample, not between-round, and the artifact says so rather than letting
a reader assume otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from theodb_bench.absent import Absent, Measured, encode, unavailable
from theodb_bench.errors import ConfigError, ErrorContext, Phase

STATISTICS_SCHEMA_VERSION: Final[int] = 1
DEFAULT_UNSTABLE_CV: Final[float] = 0.05
PERCENTILES: Final[tuple[float, ...]] = (50.0, 95.0, 99.0, 99.9)


@dataclass(frozen=True)
class LatencySummary:
    """Percentiles over the successful operations of one repetition.

    Failed and timed-out operations are counted elsewhere and deliberately do
    not appear here: folding them in as if they had succeeded would make a
    system look faster the more often it failed.
    """

    p50: Measured[float]
    p95: Measured[float]
    p99: Measured[float]
    p999: Measured[float]
    mean: Measured[float]
    stdev: Measured[float]
    minimum: Measured[float]
    maximum: Measured[float]
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "p50": encode(self.p50),
            "p95": encode(self.p95),
            "p99": encode(self.p99),
            "p999": encode(self.p999),
            "mean": encode(self.mean),
            "stdev": encode(self.stdev),
            "min": encode(self.minimum),
            "max": encode(self.maximum),
            "sample_count": self.sample_count,
        }


def summarise_latency(samples_ms: Sequence[float]) -> LatencySummary:
    """Percentiles over latency samples, in milliseconds."""
    count = len(samples_ms)
    if count == 0:
        missing = unavailable("no successful operations were recorded")
        return LatencySummary(
            p50=missing,
            p95=missing,
            p99=missing,
            p999=missing,
            mean=missing,
            stdev=missing,
            minimum=missing,
            maximum=missing,
            sample_count=0,
        )

    values = np.asarray(samples_ms, dtype=np.float64)
    p50, p95, p99, p999 = (float(np.percentile(values, q)) for q in PERCENTILES)
    # p99.9 needs roughly a thousand samples before it describes anything; below
    # that it is the maximum wearing a percentile's name.
    tail: Measured[float] = (
        p999 if count >= 1000 else unavailable(f"p99.9 needs >= 1000 samples, got {count}")
    )
    return LatencySummary(
        p50=p50,
        p95=p95,
        p99=p99,
        p999=tail,
        mean=float(values.mean()),
        stdev=float(values.std(ddof=1)) if count > 1 else unavailable("single sample"),
        minimum=float(values.min()),
        maximum=float(values.max()),
        sample_count=count,
    )


def throughput_best_of_n(round_mean_seconds: Sequence[float]) -> Measured[float]:
    """Operations per second from the fastest round (ANN-Benchmarks protocol).

    Using the fastest round rather than the average of rounds is the published
    convention; mixing the two makes numbers incomparable with every existing
    ANN result.
    """
    usable = [value for value in round_mean_seconds if value > 0]
    if not usable:
        return unavailable("no round produced a positive mean duration")
    return 1.0 / min(usable)


@dataclass(frozen=True)
class Aggregate:
    """One metric across repetitions, with the repetitions kept."""

    values: tuple[float, ...]
    median: float
    mean: float
    stdev: Measured[float]
    minimum: float
    maximum: float
    coefficient_of_variation: Measured[float]
    ci95_low: Measured[float]
    ci95_high: Measured[float]

    @property
    def repetitions(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "mean": self.mean,
            "stdev": encode(self.stdev),
            "min": self.minimum,
            "max": self.maximum,
            "coefficient_of_variation": encode(self.coefficient_of_variation),
            "ci95_low": encode(self.ci95_low),
            "ci95_high": encode(self.ci95_high),
            "repetitions": self.repetitions,
            "values": list(self.values),
        }


def aggregate(values: Sequence[float]) -> Aggregate:
    """Summarise repetitions of one metric, keeping every value."""
    if not values:
        raise ConfigError(
            "cannot aggregate zero repetitions",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    count = array.size

    stdev: Measured[float] = (
        float(array.std(ddof=1)) if count > 1 else unavailable("a single repetition has no spread")
    )
    cv: Measured[float]
    if isinstance(stdev, Absent):
        cv = unavailable("no spread to divide by the mean")
    elif mean == 0:
        cv = unavailable("mean is zero; coefficient of variation is undefined")
    else:
        cv = float(stdev / abs(mean))

    low: Measured[float]
    high: Measured[float]
    if count > 1 and not isinstance(stdev, Absent):
        # Normal approximation, honest about being one: with the handful of
        # repetitions a benchmark runs, a t-interval would be a false precision.
        margin = 1.96 * stdev / np.sqrt(count)
        low, high = mean - margin, mean + margin
    else:
        low = unavailable("a single repetition has no interval")
        high = unavailable("a single repetition has no interval")

    return Aggregate(
        values=tuple(float(v) for v in values),
        median=float(np.median(array)),
        mean=mean,
        stdev=stdev,
        minimum=float(array.min()),
        maximum=float(array.max()),
        coefficient_of_variation=cv,
        ci95_low=low,
        ci95_high=high,
    )


@dataclass(frozen=True)
class Stability:
    """Whether a point's repetitions agree well enough to be believed."""

    stable: bool
    reason: str
    threshold_cv: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "stable": self.stable,
            "reason": self.reason,
            "threshold_cv": self.threshold_cv,
        }


def assess_stability(
    aggregates: dict[str, Aggregate], threshold_cv: float = DEFAULT_UNSTABLE_CV
) -> Stability:
    """Flag a point whose repetitions disagree more than the threshold allows.

    Instability is reported, never corrected. A noisy point stays in the
    result; what changes is that the reader is told.
    """
    noisy: list[str] = []
    unknown: list[str] = []
    for name, value in aggregates.items():
        cv = value.coefficient_of_variation
        if isinstance(cv, Absent):
            unknown.append(name)
        elif cv > threshold_cv:
            noisy.append(f"{name} cv={cv:.3f}")
    if noisy:
        return Stability(False, "; ".join(sorted(noisy)), threshold_cv)
    if unknown and len(unknown) == len(aggregates):
        return Stability(
            False, f"no spread available for {', '.join(sorted(unknown))}", threshold_cv
        )
    return Stability(
        True, "all metrics within the coefficient-of-variation threshold", threshold_cv
    )


@dataclass(frozen=True)
class PointStatistics:
    """Aggregated statistics for one benchmark configuration."""

    label: str
    parameters: dict[str, Any]
    metrics: dict[str, Aggregate]
    stability: Stability

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "parameters": dict(self.parameters),
            "metrics": {name: value.as_dict() for name, value in self.metrics.items()},
            "stability": self.stability.as_dict(),
        }


def summarise_points(
    points: Sequence[tuple[str, dict[str, Any], Mapping[str, Sequence[float]]]],
    threshold_cv: float = DEFAULT_UNSTABLE_CV,
) -> list[PointStatistics]:
    """Aggregate every metric of every point across its repetitions."""
    summarised: list[PointStatistics] = []
    for label, parameters, metrics in points:
        aggregates = {name: aggregate(values) for name, values in metrics.items() if values}
        summarised.append(
            PointStatistics(
                label=label,
                parameters=parameters,
                metrics=aggregates,
                stability=assess_stability(aggregates, threshold_cv),
            )
        )
    return summarised


def statistics_payload(
    run_id: str,
    points: Sequence[PointStatistics],
    outlier_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``derived/statistics.json`` artifact."""
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "run_id": run_id,
        "outlier_policy": outlier_policy if outlier_policy is not None else {"name": "none"},
        "points": [point.as_dict() for point in points],
    }


def noise_floor(runs: Sequence[dict[str, float]]) -> dict[str, float]:
    """Coefficient of variation per metric across repeated identical runs.

    This is what a regression threshold has to be derived from. Setting a gate
    tighter than the measured noise floor produces alerts about the hardware,
    not about the change (TRD §22).
    """
    if len(runs) < 2:
        raise ConfigError(
            f"a noise floor needs at least 2 runs, got {len(runs)}",
            context=ErrorContext(phase=Phase.OFFLINE),
        )
    metrics = sorted({name for run in runs for name in run})
    floor: dict[str, float] = {}
    for name in metrics:
        values = [run[name] for run in runs if name in run]
        if len(values) < 2:
            continue
        summary = aggregate(values)
        cv = summary.coefficient_of_variation
        if not isinstance(cv, Absent):
            floor[name] = cv
    return floor
