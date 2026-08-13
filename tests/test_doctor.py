"""Preflight decides whether a host may measure; profiles decide how strictly."""

from __future__ import annotations

from pathlib import Path

from theodb_bench.doctor import (
    CHECKS,
    Check,
    DoctorReport,
    Outcome,
    check_datasets,
    render_report,
    run_doctor,
)
from theodb_bench.profiles import get_profile


def test_every_check_returns_a_known_outcome() -> None:
    for check in CHECKS:
        assert check().outcome in set(Outcome)


def test_check_ids_are_unique() -> None:
    ids = [check().id for check in CHECKS]
    assert len(ids) == len(set(ids))


def test_smoke_never_blocks_on_an_optional_capability() -> None:
    # A laptop must be able to run a smoke benchmark; it just may not publish.
    report = run_doctor(get_profile("smoke"))
    blocking_ids = {c.id for c in report.blocking}
    assert blocking_ids <= {"required_binaries"}


def test_release_is_stricter_than_smoke() -> None:
    smoke = run_doctor(get_profile("smoke"))
    release = run_doctor(get_profile("release"))
    assert len(release.blocking) >= len(smoke.blocking)


def test_a_failed_optional_check_does_not_block() -> None:
    check = Check("perf_events", Outcome.FAIL, "denied", required_for=frozenset())
    report = DoctorReport((check,), get_profile("release"))
    assert report.may_run


def test_a_mandatory_warning_blocks_the_profile_that_requires_it() -> None:
    # A release measured under frequency scaling is a methodology defect, not
    # a note to read later.
    check = Check("cpu_governor", Outcome.WARN, "powersave", required_for=frozenset({"release"}))
    assert not DoctorReport((check,), get_profile("release")).may_run
    assert DoctorReport((check,), get_profile("smoke")).may_run


def test_a_failed_mandatory_check_blocks_that_profile_only() -> None:
    check = Check("cgroup_v2", Outcome.FAIL, "not mounted", required_for=frozenset({"release"}))
    assert not DoctorReport((check,), get_profile("release")).may_run
    assert DoctorReport((check,), get_profile("smoke")).may_run


def test_unavailable_blocks_when_mandatory() -> None:
    # "We could not tell" is not permission to proceed on a release run.
    check = Check("cpu", Outcome.UNAVAILABLE, "unknown", required_for=frozenset({"release"}))
    assert not DoctorReport((check,), get_profile("release")).may_run


def test_counts_cover_every_outcome_kind() -> None:
    report = run_doctor(get_profile("smoke"))
    assert set(report.counts()) == {o.value for o in Outcome}
    assert sum(report.counts().values()) == len(report.checks)


def test_datasets_check_reports_absence_without_failing(tmp_path: Path) -> None:
    check = check_datasets(tmp_path / "never-fetched")
    assert check.outcome is Outcome.UNAVAILABLE
    assert "does not exist" in check.detail


def test_datasets_check_lists_what_is_present(tmp_path: Path) -> None:
    (tmp_path / "sift1m").mkdir()
    check = check_datasets(tmp_path)
    assert check.outcome is Outcome.PASS
    assert "sift1m" in check.detail


def test_report_serialises_for_machine_consumption() -> None:
    payload = run_doctor(get_profile("pr")).as_dict()
    assert payload["profile"] == "pr"
    assert isinstance(payload["may_run"], bool)
    assert len(payload["checks"]) == len(CHECKS)


def test_rendered_report_marks_blocking_checks() -> None:
    check = Check("cgroup_v2", Outcome.FAIL, "not mounted", required_for=frozenset({"release"}))
    rendered = render_report(DoctorReport((check,), get_profile("release")))
    assert "FAIL*" in rendered
    assert "may NOT run" in rendered


def test_rendered_report_states_permission_when_clear() -> None:
    check = Check("cpu", Outcome.PASS, "12 logical CPUs")
    rendered = render_report(DoctorReport((check,), get_profile("smoke")))
    assert "may run" in rendered
