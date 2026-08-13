"""Pareto frontiers and matched-quality comparison.

An approximate index is a trade-off, so a single number cannot describe it. The
frontier is the set of configurations no other configuration beats on every
objective at once; anything else is dominated and must not be presented as the
system's behaviour.

A headline comparison at a single quality target is allowed, but only when the
target and the selection method are stated (TRD §13.5). ``matched_quality``
returns both, or reports that no configuration reached the target -- never the
closest one pretending it did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

PARETO_SCHEMA_VERSION: Final[int] = 1

MAXIMIZE: Final[str] = "maximize"
MINIMIZE: Final[str] = "minimize"


@dataclass(frozen=True)
class Objective:
    """One axis of the trade-off."""

    metric: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in {MAXIMIZE, MINIMIZE}:
            raise ValueError(f"direction must be {MAXIMIZE!r} or {MINIMIZE!r}")

    def better(self, left: float, right: float) -> bool:
        return left > right if self.direction == MAXIMIZE else left < right

    def at_least_as_good(self, left: float, right: float) -> bool:
        return left >= right if self.direction == MAXIMIZE else left <= right

    def as_dict(self) -> dict[str, str]:
        return {"metric": self.metric, "direction": self.direction}


@dataclass(frozen=True)
class Point:
    """One configuration's position on every objective."""

    label: str
    values: dict[str, float]

    def value(self, metric: str) -> float | None:
        return self.values.get(metric)


def dominates(left: Point, right: Point, objectives: Sequence[Objective]) -> bool:
    """True when ``left`` is at least as good everywhere and better somewhere.

    A point missing a value on any objective can neither dominate nor be
    dominated on it: comparing against a number that was never measured would
    be inventing the comparison.
    """
    strictly_better = False
    for objective in objectives:
        a = left.value(objective.metric)
        b = right.value(objective.metric)
        if a is None or b is None:
            return False
        if not objective.at_least_as_good(a, b):
            return False
        if objective.better(a, b):
            strictly_better = True
    return strictly_better


def frontier(points: Sequence[Point], objectives: Sequence[Objective]) -> list[Point]:
    """The non-dominated points, in input order."""
    if not objectives:
        raise ValueError("a frontier needs at least one objective")
    return [
        candidate
        for candidate in points
        if not any(
            dominates(other, candidate, objectives) for other in points if other is not candidate
        )
    ]


def dominators(
    candidate: Point, points: Sequence[Point], objectives: Sequence[Objective]
) -> list[str]:
    """Labels of the points that dominate ``candidate``."""
    return [
        other.label
        for other in points
        if other is not candidate and dominates(other, candidate, objectives)
    ]


@dataclass(frozen=True)
class MatchedQuality:
    """A headline comparison at a stated quality target."""

    metric: str
    target: float
    method: str
    selected: str | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "target": self.target,
            "method": self.method,
            "selected": self.selected,
            "detail": self.detail,
        }


def matched_quality(
    points: Sequence[Point],
    quality_metric: str,
    target: float,
    performance_metric: str,
    performance_direction: str = MAXIMIZE,
) -> MatchedQuality:
    """Pick the best-performing configuration that actually reaches the target.

    Configurations below the target are not considered, however close. Reporting
    the nearest one as if it had reached the target is how a matched-recall
    comparison stops being matched.
    """
    objective = Objective(performance_metric, performance_direction)
    qualifying = [
        point
        for point in points
        if (value := point.value(quality_metric)) is not None and value >= target
    ]
    if not qualifying:
        observed = [value for p in points if (value := p.value(quality_metric)) is not None]
        best = max(observed) if observed else None
        detail = f"no configuration reached {quality_metric} >= {target}" + (
            f"; best observed was {best}" if best is not None else ""
        )
        return MatchedQuality(quality_metric, target, "none_available", None, detail)

    selected = qualifying[0]
    for candidate in qualifying[1:]:
        current = selected.value(performance_metric)
        contender = candidate.value(performance_metric)
        if current is None or (contender is not None and objective.better(contender, current)):
            selected = candidate

    return MatchedQuality(
        metric=quality_metric,
        target=target,
        method="nearest_at_or_above",
        selected=selected.label,
        detail=(
            f"best {performance_metric} among configurations reaching {quality_metric} >= {target}"
        ),
    )


def pareto_payload(
    points: Sequence[Point],
    objectives: Sequence[Objective],
    run_id: str | None = None,
    matched: MatchedQuality | None = None,
) -> dict[str, Any]:
    """The ``derived/pareto.json`` artifact."""
    on_frontier = {point.label for point in frontier(points, objectives)}
    payload: dict[str, Any] = {
        "schema_version": PARETO_SCHEMA_VERSION,
        "objectives": [objective.as_dict() for objective in objectives],
        "points": [
            {
                "label": point.label,
                "values": dict(point.values),
                "dominated_by": dominators(point, points, objectives),
            }
            for point in points
        ],
        "frontier": [point.label for point in points if point.label in on_frontier],
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if matched is not None:
        payload["matched_quality"] = matched.as_dict()
    return payload
