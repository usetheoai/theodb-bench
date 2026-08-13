"""Regression comparison against an accepted baseline.

Two rules decide everything here.

**Fail closed.** When the baseline is not comparable -- different benchmark
version, different hardware class, different profile -- the verdict is
INCOMPARABLE. Not PASS, not FAIL. Comparing across incompatible configurations
silently is how a real regression gets explained away as "different hardware"
and how a phantom one gets filed as a bug (TRD §22).

**A threshold that was not derived from measured variance is advisory.** Until
the benchmark's own noise floor is known, a gate cannot distinguish a
regression from the machine having a bad afternoon, and a gate that cannot do
that produces alerts about the hardware. Such gates report ADVISORY and say
where the threshold came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

REGRESSION_SCHEMA_VERSION: Final[int] = 1

HIGHER_IS_BETTER: Final[str] = "higher_is_better"
LOWER_IS_BETTER: Final[str] = "lower_is_better"

MEASURED: Final[str] = "measured_noise_floor"
DECLARED: Final[str] = "declared"
ADVISORY_SOURCE: Final[str] = "advisory"

# Fields that must match for two runs to be comparable at all. Each one changes
# what is being measured, not just how much of it.
COMPARABILITY_FIELDS: Final[tuple[str, ...]] = (
    "benchmark_id",
    "benchmark_version",
    "profile",
    "system",
    "hardware_class",
)


@dataclass(frozen=True)
class BaselineRef:
    """The accepted result a candidate is measured against."""

    run_id: str
    benchmark_id: str
    benchmark_version: int
    profile: str
    system: str
    hardware_class: str
    accepted_commit: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "profile": self.profile,
            "system": self.system,
            "hardware_class": self.hardware_class,
            "accepted_commit": self.accepted_commit,
        }


@dataclass(frozen=True)
class Candidate:
    """The run being judged."""

    run_id: str
    benchmark_id: str
    benchmark_version: int
    profile: str
    system: str
    hardware_class: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Gate:
    """One metric's regression rule."""

    metric: str
    direction: str
    max_regression_pct: float | None = None
    max_absolute_regression: float | None = None
    threshold_source: str = ADVISORY_SOURCE

    def __post_init__(self) -> None:
        if self.direction not in {HIGHER_IS_BETTER, LOWER_IS_BETTER}:
            raise ValueError(f"unknown direction {self.direction!r}")
        if self.threshold_source not in {MEASURED, DECLARED, ADVISORY_SOURCE}:
            raise ValueError(f"unknown threshold source {self.threshold_source!r}")


@dataclass(frozen=True)
class GateOutcome:
    """How one gate judged the candidate."""

    gate: Gate
    candidate: float | None
    baseline: float | None
    delta_pct: float | None
    outcome: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.gate.metric,
            "direction": self.gate.direction,
            "candidate": self.candidate,
            "baseline": self.baseline,
            "delta_pct": self.delta_pct,
            "max_regression_pct": self.gate.max_regression_pct,
            "max_absolute_regression": self.gate.max_absolute_regression,
            "threshold_source": self.gate.threshold_source,
            "outcome": self.outcome,
            "detail": self.detail,
        }


def check_comparability(
    candidate: Candidate, baseline: BaselineRef | None
) -> tuple[bool, list[dict[str, Any]]]:
    """Whether these two runs may be compared at all."""
    if baseline is None:
        return False, [
            {"field": "baseline", "matches": False, "candidate": candidate.run_id, "baseline": None}
        ]
    checks: list[dict[str, Any]] = []
    comparable = True
    for name in COMPARABILITY_FIELDS:
        left = getattr(candidate, name)
        right = getattr(baseline, name)
        matches = left == right
        comparable = comparable and matches
        checks.append({"field": name, "matches": matches, "candidate": left, "baseline": right})
    return comparable, checks


def _relative_change(candidate: float, baseline: float, direction: str) -> float | None:
    """Percentage change, signed so that positive always means worse."""
    if baseline == 0:
        return None
    change = (candidate - baseline) / abs(baseline) * 100.0
    return -change if direction == HIGHER_IS_BETTER else change


def evaluate_gate(gate: Gate, candidate: Candidate, baseline: BaselineRef) -> GateOutcome:
    """Judge one metric.

    A metric missing from either side is UNAVAILABLE rather than PASS: an
    absent measurement cannot demonstrate the absence of a regression.
    """
    left = candidate.metrics.get(gate.metric)
    right = baseline.metrics.get(gate.metric)
    if left is None or right is None:
        missing = "candidate" if left is None else "baseline"
        return GateOutcome(
            gate,
            left,
            right,
            None,
            "UNAVAILABLE",
            f"{gate.metric} not measured on the {missing}; "
            "an absent measurement cannot show the absence of a regression",
        )

    regression_pct = _relative_change(left, right, gate.direction)
    absolute = (right - left) if gate.direction == HIGHER_IS_BETTER else (left - right)

    breached: list[str] = []
    if (
        gate.max_regression_pct is not None
        and regression_pct is not None
        and regression_pct > gate.max_regression_pct
    ):
        breached.append(f"{regression_pct:.2f}% worse, budget {gate.max_regression_pct:.2f}%")
    if gate.max_absolute_regression is not None and absolute > gate.max_absolute_regression:
        breached.append(
            f"{absolute:.6g} worse in absolute terms, budget {gate.max_absolute_regression:.6g}"
        )

    if not breached:
        return GateOutcome(
            gate, left, right, regression_pct, "PASS", f"{gate.metric} within budget"
        )

    # A threshold nobody derived from measured variance cannot tell a
    # regression from noise, so it advises rather than fails.
    outcome = "FAIL" if gate.threshold_source == MEASURED else "ADVISORY"
    suffix = (
        ""
        if gate.threshold_source == MEASURED
        else " (threshold not derived from a measured noise floor, so advisory)"
    )
    return GateOutcome(gate, left, right, regression_pct, outcome, "; ".join(breached) + suffix)


def gates_from_noise_floor(
    floor: dict[str, float], directions: dict[str, str], multiplier: float = 3.0
) -> list[Gate]:
    """Derive gates from measured variance.

    The budget is a multiple of the metric's own coefficient of variation, so a
    gate is looser on a metric the hardware cannot measure tightly. Setting a
    budget below the noise floor produces alerts about the machine.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    return [
        Gate(
            metric=metric,
            direction=directions.get(metric, HIGHER_IS_BETTER),
            max_regression_pct=cv * 100.0 * multiplier,
            threshold_source=MEASURED,
        )
        for metric, cv in sorted(floor.items())
    ]


def compare(
    candidate: Candidate,
    baseline: BaselineRef | None,
    gates: list[Gate],
    noise_floor: dict[str, float] | None = None,
    noise_floor_runs: int | None = None,
) -> dict[str, Any]:
    """The ``derived/regression.json`` artifact."""
    comparable, checks = check_comparability(candidate, baseline)

    payload: dict[str, Any] = {
        "schema_version": REGRESSION_SCHEMA_VERSION,
        "candidate_run_id": candidate.run_id,
        "baseline": baseline.as_dict() if baseline is not None else None,
        "comparability": {"comparable": comparable, "checks": checks},
        "gates": [],
        "verdict": "INCOMPARABLE",
    }

    if noise_floor and noise_floor_runs:
        payload["noise_floor"] = {
            "source_runs": noise_floor_runs,
            "metric_cv": dict(noise_floor),
        }

    if not comparable or baseline is None:
        mismatched = [c["field"] for c in checks if not c["matches"]]
        payload["notes"] = (
            "Baseline is not comparable ("
            + ", ".join(mismatched)
            + "). Comparing anyway would let a real regression be explained away "
            "and a phantom one be filed as a bug."
        )
        return payload

    outcomes = [evaluate_gate(gate, candidate, baseline) for gate in gates]
    payload["gates"] = [outcome.as_dict() for outcome in outcomes]

    if any(outcome.outcome == "FAIL" for outcome in outcomes):
        payload["verdict"] = "FAIL"
    elif any(outcome.outcome == "ADVISORY" for outcome in outcomes):
        payload["verdict"] = "ADVISORY"
    elif any(outcome.outcome == "UNAVAILABLE" for outcome in outcomes):
        # Some metric could not be judged. Reporting PASS would claim more than
        # was checked.
        payload["verdict"] = "ADVISORY"
        payload["notes"] = "Some gates could not be evaluated; see their detail."
    else:
        payload["verdict"] = "PASS"
    return payload
