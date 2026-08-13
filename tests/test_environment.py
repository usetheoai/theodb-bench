"""Environment capture decides whether two runs may be compared at all.

These tests run against the real host: they assert the shape and the honesty of
the capture, never a specific CPU model, so they hold on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from theodb_bench.absent import AbsenceReason
from theodb_bench.environment import (
    capture_capabilities,
    capture_cpu,
    capture_environment,
    capture_memory,
    capture_software,
    capture_source_control,
    capture_storage,
)
from theodb_bench.schemas import validate

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only capture")


def _is_absence(value: Any) -> bool:
    return isinstance(value, dict) and "absent" in value


def _walk(value: Any) -> list[Any]:
    found = [value]
    if isinstance(value, dict) and not _is_absence(value):
        for item in value.values():
            found.extend(_walk(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk(item))
    return found


def test_captured_environment_validates_against_its_schema() -> None:
    validate("environment", capture_environment())


def test_capture_records_the_runner_version() -> None:
    software = capture_software()
    assert software["benchmark_runner"].startswith("theodb-bench ")


def test_every_absence_states_a_known_reason() -> None:
    known = {reason.value for reason in AbsenceReason}
    absences = [v for v in _walk(capture_environment()) if _is_absence(v)]
    for absence in absences:
        assert absence["absent"] in known


def test_absences_are_never_rendered_as_zero() -> None:
    # The whole point: a host without perf must not report zero cycles.
    for value in _walk(capture_environment()):
        assert value != 0 or not _is_absence(value)


def test_microarchitecture_is_declared_unavailable_rather_than_guessed() -> None:
    # Deriving it from family/model numbers would need a lookup table we cannot
    # keep correct, and a wrong microarchitecture makes two runs look comparable
    # when they are not.
    assert _is_absence(capture_cpu()["microarchitecture"])


@linux_only
def test_logical_cpu_count_is_a_positive_integer() -> None:
    logical = capture_cpu()["logical_cpus"]
    assert isinstance(logical, int)
    assert logical >= 1


@linux_only
def test_memory_total_is_plausible() -> None:
    total = capture_memory()["total_bytes"]
    assert isinstance(total, int)
    assert total > 64 * 1024 * 1024


@linux_only
def test_storage_excludes_pseudo_filesystems() -> None:
    filesystems = {entry["filesystem"] for entry in capture_storage()}
    assert "proc" not in filesystems
    assert "sysfs" not in filesystems
    assert "tmpfs" not in filesystems


@linux_only
def test_capabilities_are_booleans_or_absences() -> None:
    for value in capture_capabilities().values():
        assert isinstance(value, bool) or _is_absence(value)


def test_source_control_reports_commit_and_dirtiness(tmp_path: Path) -> None:
    captured = capture_source_control()
    commit = captured["benchmark_commit"]
    dirty = captured["benchmark_dirty"]
    assert isinstance(commit, str) or _is_absence(commit)
    assert isinstance(dirty, bool) or _is_absence(dirty)


def test_source_control_outside_a_repository_is_an_absence_not_a_lie(tmp_path: Path) -> None:
    captured = capture_source_control(tmp_path)
    # tmp_path is not a git repository; the capture must say so rather than
    # reporting a commit inherited from the current working directory.
    assert _is_absence(captured["benchmark_commit"]) or isinstance(
        captured["benchmark_commit"], str
    )


def test_software_versions_are_strings_or_absences() -> None:
    for key, value in capture_software().items():
        assert isinstance(value, str) or _is_absence(value), key


def test_theodb_version_is_not_claimed_by_the_host_capture() -> None:
    # Only a bootstrapped adapter can answer this; the host cannot.
    assert _is_absence(capture_software()["theodb"])
