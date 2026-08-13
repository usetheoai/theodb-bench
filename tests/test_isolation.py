"""Isolation matters less than detecting that isolation did not hold.

The escape tests spawn real processes: a unit test with a mocked procfs would
prove only that the mock behaves as written.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from theodb_bench.absent import Absent
from theodb_bench.errors import IsolationError, ProcessEscapedError
from theodb_bench.isolation import (
    ExclusiveBenchmarkLock,
    IsolationPlan,
    apply_cpu_affinity,
    apply_isolation,
    assert_no_escapes,
    find_escapes,
    format_cpu_set,
    online_cpus,
    parse_cpu_set,
    probe_cgroup_support,
    process_affinity,
    process_tree,
)

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only")


# ------------------------------------------------------------------- cpu sets


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0", {0}),
        ("0-3", {0, 1, 2, 3}),
        ("0-1,4", {0, 1, 4}),
        ("2, 5 , 7-8", {2, 5, 7, 8}),
    ],
)
def test_cpu_sets_parse(spec: str, expected: set[int]) -> None:
    assert parse_cpu_set(spec) == expected


@pytest.mark.parametrize("spec", ["", "  ", "a", "3-1", "0-", "1-x"])
def test_malformed_cpu_sets_are_rejected(spec: str) -> None:
    with pytest.raises(IsolationError):
        parse_cpu_set(spec)


@pytest.mark.parametrize(
    ("cpus", "rendered"),
    [({0}, "0"), ({0, 1, 2}, "0-2"), ({0, 2}, "0,2"), ({0, 1, 4, 5, 9}, "0-1,4-5,9")],
)
def test_cpu_sets_render_compactly(cpus: set[int], rendered: str) -> None:
    assert format_cpu_set(frozenset(cpus)) == rendered


def test_cpu_set_round_trips() -> None:
    assert parse_cpu_set(format_cpu_set(frozenset({0, 1, 4, 7}))) == {0, 1, 4, 7}


# -------------------------------------------------------------- process trees


@linux_only
def test_process_tree_includes_a_spawned_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if child.pid in process_tree(os.getpid()):
                break
            time.sleep(0.05)
        assert child.pid in process_tree(os.getpid())
    finally:
        child.kill()
        child.wait(timeout=5)


@linux_only
def test_process_tree_drops_a_process_that_exited() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    assert child.pid not in process_tree(os.getpid())


@linux_only
def test_affinity_of_this_process_is_readable() -> None:
    affinity = process_affinity(os.getpid())
    assert not isinstance(affinity, Absent)
    assert affinity == online_cpus()


def test_affinity_of_a_missing_process_is_an_absence_not_an_empty_set() -> None:
    # An empty set would read as "allowed nowhere", which is a claim.
    assert isinstance(process_affinity(2**31 - 1), Absent)


# ------------------------------------------------------------------- escapes


@linux_only
def test_a_child_outside_the_declared_cpu_set_is_detected() -> None:
    available = sorted(online_cpus())
    if len(available) < 2:
        pytest.skip("needs at least two CPUs to have somewhere to escape to")
    declared = frozenset({available[0]})

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        # The child inherits this process's full affinity, which is wider than
        # the declared set: exactly the escape the runner must catch.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if child.pid in process_tree(os.getpid()):
                break
            time.sleep(0.05)

        escapes = find_escapes(os.getpid(), declared)
        escaped_pids = {escape.pid for escape in escapes}
        assert child.pid in escaped_pids

        with pytest.raises(ProcessEscapedError, match="outside the declared CPU set"):
            assert_no_escapes(os.getpid(), declared)
    finally:
        child.kill()
        child.wait(timeout=5)


@linux_only
def test_a_child_pinned_inside_the_declared_set_is_not_an_escape() -> None:
    available = sorted(online_cpus())
    declared = frozenset(available)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if child.pid in process_tree(os.getpid()):
                break
            time.sleep(0.05)
        assert not find_escapes(os.getpid(), declared)
    finally:
        child.kill()
        child.wait(timeout=5)


@linux_only
def test_escape_records_carry_enough_context_to_act_on() -> None:
    available = sorted(online_cpus())
    if len(available) < 2:
        pytest.skip("needs at least two CPUs")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if child.pid in process_tree(os.getpid()):
                break
            time.sleep(0.05)
        escapes = [
            e for e in find_escapes(os.getpid(), frozenset({available[0]})) if e.pid == child.pid
        ]
        assert escapes
        record = escapes[0].as_dict()
        assert record["pid"] == child.pid
        assert "python" in record["command"].lower()
        assert record["declared"] == format_cpu_set(frozenset({available[0]}))
    finally:
        child.kill()
        child.wait(timeout=5)


# ---------------------------------------------------------------------- lock


_SECOND_HOLDER = """
import sys
from pathlib import Path

from theodb_bench.errors import IsolationError
from theodb_bench.isolation import ExclusiveBenchmarkLock

try:
    ExclusiveBenchmarkLock(Path(sys.argv[1])).acquire()
except IsolationError:
    sys.exit(7)
sys.exit(0)
"""


def test_exclusive_lock_blocks_a_second_holder(tmp_path: Path) -> None:
    # Two benchmarks sharing a host measure each other. flock is per open file
    # description, so a second acquisition inside this process would succeed;
    # the contention that matters is between processes, and that is what runs.
    lock_path = tmp_path / "bench.lock"
    with ExclusiveBenchmarkLock(lock_path):
        result = subprocess.run(
            [sys.executable, "-c", _SECOND_HOLDER, str(lock_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 7, result.stderr


def test_lock_is_released_on_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / "bench.lock"
    with ExclusiveBenchmarkLock(lock_path):
        pass
    with ExclusiveBenchmarkLock(lock_path):
        pass


# ------------------------------------------------------------------ applying


def test_pinning_to_an_unavailable_cpu_is_an_error_not_a_silent_noop() -> None:
    impossible = frozenset({max(online_cpus()) + 100})
    with pytest.raises(IsolationError, match="not available"):
        apply_cpu_affinity(os.getpid(), impossible)


@linux_only
def test_applying_a_cpu_set_reports_success_and_is_observable() -> None:
    original = online_cpus()
    target = frozenset({sorted(original)[0]})
    try:
        assert apply_cpu_affinity(os.getpid(), target) is True
        assert process_affinity(os.getpid()) == target
    finally:
        os.sched_setaffinity(0, original)


def test_unenforceable_controls_are_reported_as_absent_never_as_applied() -> None:
    # This is the invariant that keeps a release run honest: what could not be
    # enforced must not be recorded as enforced.
    applied = apply_isolation(IsolationPlan(memory_bytes=1 << 30, numa_node=0))
    assert isinstance(applied.memory_limit_applied, Absent)
    assert isinstance(applied.numa_applied, Absent)
    assert applied.notes


def test_applied_isolation_serialises_declared_and_achieved() -> None:
    payload = apply_isolation(
        IsolationPlan(cpu_set=frozenset({sorted(online_cpus())[0]}))
    ).as_dict()
    assert "declared" in payload
    assert "cpu_affinity_applied" in payload


def test_cgroup_probe_states_why_it_is_unusable_when_it_is() -> None:
    support = probe_cgroup_support()
    assert support.detail
    if not support.usable:
        assert "not writable" in support.detail or "not mounted" in support.detail
