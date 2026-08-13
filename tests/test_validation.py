"""A run is invalidated by how it was executed, never by what it measured."""

from __future__ import annotations

from theodb_bench.absent import unavailable
from theodb_bench.profiles import get_profile
from theodb_bench.schemas import validate as validate_schema
from theodb_bench.validation import (
    RunObservations,
    ValidationPolicy,
    build_checks,
    validate_run,
)


def _clean(**overrides: object) -> RunObservations:
    base: dict[str, object] = {
        "observed_operations": 100_000,
        "expected_operations": 100_000,
        "repetitions_declared": 3,
        "repetitions_completed": 3,
    }
    base.update(overrides)
    return RunObservations(**base)  # type: ignore[arg-type]


def _outcome(payload: dict[str, object], check_id: str) -> str:
    checks = payload["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if check["id"] == check_id:
            return str(check["outcome"])
    raise AssertionError(f"no check {check_id!r}")


# --------------------------------------------------------------------- shape


def test_result_conforms_to_the_validation_schema() -> None:
    validate_schema("validation", validate_run(_clean(), get_profile("pr")))


def test_a_clean_run_is_valid() -> None:
    assert validate_run(_clean(), get_profile("pr"))["status"] == "VALID"


def test_check_ids_are_unique() -> None:
    ids = [c.id for c in build_checks(_clean(), get_profile("release"))]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------- invalidation


def test_a_crashed_system_invalidates_the_run() -> None:
    payload = validate_run(_clean(sut_crashed=True), get_profile("smoke"))
    assert payload["status"] == "INVALID"
    assert "sut_alive" in payload["invalidated_by"]


def test_a_crashed_client_invalidates_the_run() -> None:
    assert validate_run(_clean(client_crashed=True), get_profile("smoke"))["status"] == "INVALID"


def test_missing_operations_invalidate_the_run() -> None:
    payload = validate_run(_clean(observed_operations=90_000), get_profile("pr"))
    assert payload["status"] == "INVALID"
    assert "operation_count" in payload["invalidated_by"]


def test_declared_tolerance_is_honoured() -> None:
    payload = validate_run(
        _clean(observed_operations=99_500),
        get_profile("pr"),
        ValidationPolicy(operation_count_tolerance=0.01),
    )
    assert payload["status"] == "VALID"


def test_an_unfinished_repetition_invalidates_the_run() -> None:
    payload = validate_run(_clean(repetitions_completed=2), get_profile("pr"))
    assert "repetitions_completed" in payload["invalidated_by"]


def test_timeouts_above_policy_invalidate_the_run() -> None:
    payload = validate_run(_clean(timeouts=1_000), get_profile("pr"))
    assert "timeout_rate" in payload["invalidated_by"]


def test_errors_above_policy_invalidate_the_run() -> None:
    payload = validate_run(_clean(errors=5), get_profile("pr"))
    assert "error_rate" in payload["invalidated_by"]


def test_invalid_result_rows_invalidate_the_run() -> None:
    payload = validate_run(_clean(invalid_results=1), get_profile("smoke"))
    assert "result_integrity" in payload["invalidated_by"]


def test_an_oom_termination_invalidates_the_run() -> None:
    payload = validate_run(_clean(oom_observed=True), get_profile("smoke"))
    assert "no_oom" in payload["invalidated_by"]


def test_an_extended_warmup_invalidates_the_run() -> None:
    # Extending warm-up until the desired number appears is the classic way to
    # produce a favourable result honestly-looking.
    payload = validate_run(_clean(warmup_honoured=False), get_profile("smoke"))
    assert "warmup_policy" in payload["invalidated_by"]


# ------------------------------------------------------------------ isolation


def test_an_escaped_process_invalidates_a_gated_profile() -> None:
    payload = validate_run(_clean(escaped_processes=(4242,)), get_profile("pr"))
    assert payload["status"] == "INVALID"
    assert "process_containment" in payload["invalidated_by"]


def test_an_escaped_process_is_recorded_but_not_fatal_for_smoke() -> None:
    # Smoke does not require isolation, so the finding is reported rather than
    # fatal -- but it is still reported.
    payload = validate_run(_clean(escaped_processes=(4242,)), get_profile("smoke"))
    assert payload["status"] == "VALID"
    assert _outcome(payload, "process_containment") == "FAIL"


def test_exceeding_the_cpu_allocation_invalidates_a_gated_profile() -> None:
    payload = validate_run(_clean(cpu_limit_respected=False), get_profile("nightly"))
    assert "cpu_limit" in payload["invalidated_by"]


def test_an_unobserved_limit_is_unavailable_not_pass() -> None:
    # Not having looked is not the same as having looked and found nothing.
    payload = validate_run(
        _clean(memory_limit_respected=unavailable("cgroup accounting off")),
        get_profile("release"),
    )
    assert _outcome(payload, "memory_limit") == "UNAVAILABLE"
    assert payload["status"] == "INVALID"


# -------------------------------------------------------------------- quality


def test_an_approximate_run_without_quality_is_invalid() -> None:
    payload = validate_run(_clean(quality_reported=False), get_profile("smoke"))
    assert "quality_reported" in payload["invalidated_by"]


def test_quality_is_not_demanded_where_the_workload_has_none() -> None:
    payload = validate_run(
        _clean(quality_required=False, quality_reported=False), get_profile("smoke")
    )
    assert payload["status"] == "VALID"
    assert all(c["id"] != "quality_reported" for c in payload["checks"])


# ---------------------------------------------------------------- provenance


def test_a_dirty_tree_invalidates_a_release_but_not_a_smoke_run() -> None:
    assert validate_run(_clean(dirty_source_tree=True), get_profile("release"))["status"] == (
        "INVALID"
    )
    assert validate_run(_clean(dirty_source_tree=True), get_profile("smoke"))["status"] == "VALID"


def test_incomplete_telemetry_invalidates_only_a_release() -> None:
    assert (
        "telemetry_complete"
        in validate_run(_clean(telemetry_complete=False), get_profile("release"))["invalidated_by"]
    )
    assert (
        validate_run(_clean(telemetry_complete=False), get_profile("nightly"))["status"] == "VALID"
    )


# --------------------------------------------------------------- exploratory


def test_a_research_run_is_exploratory_even_when_clean() -> None:
    payload = validate_run(_clean(), get_profile("research"))
    assert payload["status"] == "EXPLORATORY"
    assert "not evidence" in str(payload["notes"])


def test_a_broken_research_run_is_invalid_not_exploratory() -> None:
    assert validate_run(_clean(sut_crashed=True), get_profile("research"))["status"] == "INVALID"


# ---------------------------------------------------------------- neutrality


def test_validation_never_consults_a_measured_value() -> None:
    # The observation record carries no throughput, latency or recall at all,
    # which is what makes it impossible for a good number to rescue a broken
    # run or a bad number to condemn a sound one.
    fields = set(RunObservations.__dataclass_fields__)
    forbidden = {"throughput", "latency", "recall", "qps", "score", "ndcg"}
    assert not any(any(term in name for term in forbidden) for name in fields)
