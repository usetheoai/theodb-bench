"""Protocol validation of a completed run.

Invalidation is based on protocol criteria, never on whether the number looks
good or bad (TRD 6.9). Nothing in this module may consult a measured value to
decide whether the run counts; it consults only whether the run was executed
the way it was declared.

The whole judgement is a pure function of observed facts, so it can be tested
without executing a benchmark -- and so it cannot quietly depend on anything
else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from theodb_bench.abort import AbortKind
from theodb_bench.absent import Absent, Measured, is_present
from theodb_bench.profiles import Profile

VALIDATION_SCHEMA_VERSION: Final[int] = 1

DEFAULT_MAX_TIMEOUT_RATE: Final[float] = 0.001
DEFAULT_MAX_ERROR_RATE: Final[float] = 0.0


@dataclass(frozen=True)
class RunObservations:
    """What was observed about a run, independent of what it measured."""

    observed_operations: int
    expected_operations: int | None = None
    repetitions_declared: int = 1
    repetitions_completed: int = 1
    timeouts: int = 0
    errors: int = 0
    invalid_results: int = 0
    sut_crashed: bool = False
    run_refused: bool = False
    budget_exceeded: bool = False
    client_crashed: bool = False
    escaped_processes: tuple[int, ...] = ()
    cpu_limit_respected: Measured[bool] = True
    memory_limit_respected: Measured[bool] = True
    oom_observed: bool = False
    quality_reported: bool = True
    quality_required: bool = True
    telemetry_complete: Measured[bool] = True
    warmup_honoured: bool = True
    dirty_source_tree: Measured[bool] = False

    @property
    def total_operations(self) -> int:
        return self.observed_operations + self.timeouts + self.errors

    @property
    def timeout_rate(self) -> float:
        total = self.total_operations
        return self.timeouts / total if total else 0.0

    @property
    def error_rate(self) -> float:
        total = self.total_operations
        return self.errors / total if total else 0.0


@dataclass(frozen=True)
class ValidationPolicy:
    """Thresholds a run must satisfy. Declared per benchmark, not per result."""

    max_timeout_rate: float = DEFAULT_MAX_TIMEOUT_RATE
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE
    operation_count_tolerance: float = 0.0
    """Fraction by which observed operations may differ from expected."""


@dataclass
class _Check:
    id: str
    outcome: str
    required: bool
    description: str = ""
    detail: str | None = None
    observed: Any = None
    expected: Any = None
    _extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "outcome": self.outcome,
            "required": self.required,
        }
        if self.description:
            payload["description"] = self.description
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.observed is not None:
            payload["observed"] = self.observed
        if self.expected is not None:
            payload["expected"] = self.expected
        return payload


def _measured_check(
    check_id: str,
    value: Measured[bool],
    *,
    required: bool,
    description: str,
    failure_detail: str,
) -> _Check:
    """Turn a tri-state observation into a check.

    An absent observation is UNAVAILABLE, not PASS: not having looked is not
    the same as having looked and found nothing wrong.
    """
    if isinstance(value, Absent):
        return _Check(check_id, "UNAVAILABLE", required, description, detail=str(value))
    if value:
        return _Check(check_id, "PASS", required, description)
    return _Check(check_id, "FAIL", required, description, detail=failure_detail)


def _operation_count_check(obs: RunObservations, policy: ValidationPolicy) -> _Check:
    if obs.expected_operations is None:
        return _Check(
            "operation_count",
            "UNAVAILABLE",
            required=False,
            description="Observed operations match the declared window.",
            detail="benchmark declared no expected operation count",
            observed=obs.observed_operations,
        )
    expected = obs.expected_operations
    tolerance = expected * policy.operation_count_tolerance
    within = abs(obs.observed_operations - expected) <= tolerance
    return _Check(
        "operation_count",
        "PASS" if within else "FAIL",
        required=True,
        description="Observed operations match the declared window.",
        detail=None if within else "operation count outside declared tolerance",
        observed=obs.observed_operations,
        expected=expected,
    )


def build_checks(
    obs: RunObservations, profile: Profile, policy: ValidationPolicy | None = None
) -> list[_Check]:
    """Every protocol check for a run, in report order."""
    rules = policy if policy is not None else ValidationPolicy()
    checks: list[_Check] = [
        _Check(
            "sut_alive",
            "FAIL" if obs.sut_crashed else "PASS",
            required=True,
            description="System under test did not crash.",
            detail=AbortKind.CRASHED.detail if obs.sut_crashed else None,
        ),
        _Check(
            "run_not_refused",
            "FAIL" if obs.run_refused else "PASS",
            required=True,
            description="The harness did not refuse to measure.",
            detail=AbortKind.REFUSED.detail if obs.run_refused else None,
        ),
        _Check(
            "within_time_budget",
            "FAIL" if obs.budget_exceeded else "PASS",
            required=True,
            description="No statement was cancelled by a harness time budget.",
            detail=AbortKind.BUDGET_EXCEEDED.detail if obs.budget_exceeded else None,
        ),
        _Check(
            "client_alive",
            "FAIL" if obs.client_crashed else "PASS",
            required=True,
            description="Benchmark client did not crash.",
            detail="benchmark client crashed during the run" if obs.client_crashed else None,
        ),
        _operation_count_check(obs, rules),
        _Check(
            "repetitions_completed",
            "PASS" if obs.repetitions_completed >= obs.repetitions_declared else "FAIL",
            required=True,
            description="Every declared repetition ran.",
            observed=obs.repetitions_completed,
            expected=obs.repetitions_declared,
        ),
        _Check(
            "timeout_rate",
            "PASS" if obs.timeout_rate <= rules.max_timeout_rate else "FAIL",
            required=True,
            description="Timeout rate within policy.",
            observed=round(obs.timeout_rate, 6),
            expected=rules.max_timeout_rate,
        ),
        _Check(
            "error_rate",
            "PASS" if obs.error_rate <= rules.max_error_rate else "FAIL",
            required=True,
            description="Error rate within policy.",
            observed=round(obs.error_rate, 6),
            expected=rules.max_error_rate,
        ),
        _Check(
            "result_integrity",
            "PASS" if obs.invalid_results == 0 else "FAIL",
            required=True,
            description="No invalid result rows were returned.",
            observed=obs.invalid_results,
            expected=0,
        ),
        _Check(
            "warmup_policy",
            "PASS" if obs.warmup_honoured else "FAIL",
            required=True,
            description="Warm-up followed the declared policy.",
            detail=None if obs.warmup_honoured else "warm-up deviated from the declared policy",
        ),
        _Check(
            "process_containment",
            "PASS" if not obs.escaped_processes else "FAIL",
            required=profile.isolation_required,
            description="No process escaped the declared resource controls.",
            detail=(
                f"{len(obs.escaped_processes)} process(es) outside isolation: "
                f"{', '.join(str(pid) for pid in obs.escaped_processes)}"
                if obs.escaped_processes
                else None
            ),
        ),
        _measured_check(
            "cpu_limit",
            obs.cpu_limit_respected,
            required=profile.isolation_required,
            description="Declared CPU allocation was respected.",
            failure_detail="observed CPU usage exceeded the declared allocation",
        ),
        _measured_check(
            "memory_limit",
            obs.memory_limit_respected,
            required=profile.isolation_required,
            description="Declared memory bound was respected.",
            failure_detail="observed memory usage exceeded the declared bound",
        ),
        _Check(
            "no_oom",
            "FAIL" if obs.oom_observed else "PASS",
            required=True,
            description="No out-of-memory termination occurred.",
            detail="an OOM termination was observed" if obs.oom_observed else None,
        ),
        _measured_check(
            "telemetry_complete",
            obs.telemetry_complete,
            required=profile.name.value in {"release"},
            description="Declared telemetry was fully collected.",
            failure_detail="a declared collector produced no data",
        ),
    ]

    if obs.quality_required:
        checks.append(
            _Check(
                "quality_reported",
                "PASS" if obs.quality_reported else "FAIL",
                required=True,
                description="Quality metric was computed for an approximate workload.",
                detail=(
                    None
                    if obs.quality_reported
                    else "approximate retrieval reported without a quality axis"
                ),
            )
        )

    checks.append(_clean_tree_check(obs, profile))
    return checks


def _clean_tree_check(obs: RunObservations, profile: Profile) -> _Check:
    dirty = obs.dirty_source_tree
    if isinstance(dirty, Absent):
        return _Check(
            "clean_source_tree",
            "UNAVAILABLE",
            required=profile.dirty_tree_invalidates,
            description="Benchmark source tree was committed.",
            detail=str(dirty),
        )
    return _Check(
        "clean_source_tree",
        "FAIL" if dirty else "PASS",
        required=profile.dirty_tree_invalidates,
        description="Benchmark source tree was committed.",
        detail="uncommitted changes in the benchmark tree" if dirty else None,
    )


def validate_run(
    obs: RunObservations, profile: Profile, policy: ValidationPolicy | None = None
) -> dict[str, Any]:
    """Judge a run and produce the validation artifact.

    Returns a payload conforming to the validation schema. The status is
    derived from the checks alone.
    """
    checks = build_checks(obs, profile, policy)
    failed_required = [c.id for c in checks if c.required and c.outcome in {"FAIL", "UNAVAILABLE"}]

    if failed_required:
        status = "INVALID"
    elif not profile.publishable and profile.name.value == "research":
        # Research runs may use non-frozen parameters, so even a technically
        # clean one is not evidence for a claim.
        status = "EXPLORATORY"
    else:
        status = "VALID"

    payload: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": status,
        "checks": [c.as_dict() for c in checks],
        "invalidated_by": failed_required,
    }
    if status == "EXPLORATORY":
        payload["notes"] = (
            "Research profile: parameters are not frozen, so this result is not evidence "
            "for a published claim."
        )
    return payload


def is_present_bool(value: Measured[bool]) -> bool:
    """Whether a tri-state observation actually holds a value."""
    return is_present(value)
