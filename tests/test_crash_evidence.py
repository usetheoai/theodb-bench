"""Telling "the server restarted" apart from "the server is unreachable".

Both surface as `sut_alive` FAIL, and they are different findings. Measured
2026-08-17: a 20 000 000-vector index build was OOM-killed, PostgreSQL recovered
in 2.98 s, and the run reported only "crashed or became unreachable". Separating
the two took reading `docker logs` and `dmesg` by hand — outside the bundle,
unavailable to anyone reading it later.

`pg_postmaster_start_time()` settles it with one query and no privileged access:
a start time *later* than the run's start means the server went down and came
back, which is a defect in the system under test. An unchanged start time with a
dead connection means the path to it broke, which is usually the environment.

Deliberately not attempted here: reading the server's own log, or the kernel's.
Both need access the harness has no claim to, and a benchmark that silently
requires root on the database host is a benchmark most people cannot run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from theodb_bench.abort import AbortKind, classify_abort
from theodb_bench.adapters.base import SystemAdapter
from theodb_bench.errors import ErrorContext, Phase, SystemUnavailableError

RUN_START = datetime(2026, 8, 17, 22, 17, 31, tzinfo=timezone.utc)


class _Adapter:
    """Answers the restart probe the way a real adapter would."""

    system_id = "theodb"

    def __init__(self, start_time: datetime | None, raises: Exception | None = None) -> None:
        self._start_time = start_time
        self._raises = raises
        self.probes = 0

    def postmaster_start_time(self) -> datetime | None:
        self.probes += 1
        if self._raises is not None:
            raise self._raises
        return self._start_time


def _unavailable() -> SystemUnavailableError:
    return SystemUnavailableError(
        "could not connect to theodb",
        context=ErrorContext(phase=Phase.MEASUREMENT, system="theodb"),
    )


def test_a_restart_after_the_run_started_is_reported_as_a_restart() -> None:
    """The OOM case: the server died and came back. That is a finding about the
    system under test, and the bundle should say so without anyone reading a
    kernel log."""
    from theodb_bench.abort import describe_unavailability

    adapter = _Adapter(start_time=RUN_START + timedelta(minutes=16))

    detail = describe_unavailability(adapter, run_started_at=RUN_START)

    assert "restart" in detail.lower()
    assert "16" in detail or "22:33" in detail


def test_a_start_time_older_than_the_run_means_the_path_broke_not_the_server() -> None:
    """The server never went down, so the connection did. Reporting that as a
    crash of the system under test would blame the wrong component."""
    from theodb_bench.abort import describe_unavailability

    adapter = _Adapter(start_time=RUN_START - timedelta(hours=7))

    detail = describe_unavailability(adapter, run_started_at=RUN_START)

    assert "did not restart" in detail.lower()


def test_a_probe_that_cannot_connect_says_so_rather_than_guessing() -> None:
    """If the server is still unreachable, the probe answers nothing — and
    nothing is the honest answer, not a default to either explanation."""
    from theodb_bench.abort import describe_unavailability

    adapter = _Adapter(start_time=None, raises=_unavailable())

    detail = describe_unavailability(adapter, run_started_at=RUN_START)

    assert "could not" in detail.lower()
    assert "restart" not in detail.lower().replace("restarted", "")


def test_an_adapter_without_the_probe_is_not_an_error() -> None:
    """Only the PostgreSQL family can answer this. An adapter that cannot is
    silent about it rather than failing the run for a diagnostic."""
    from theodb_bench.abort import describe_unavailability

    class _Bare:
        system_id = "something-else"

    detail = describe_unavailability(_Bare(), run_started_at=RUN_START)

    assert detail == ""


def test_the_abort_kind_is_unchanged_by_the_diagnosis() -> None:
    """The diagnosis explains a crash; it never reclassifies one. A restart is
    still CRASHED, and the run is still invalid."""
    kind = classify_abort(_unavailable())

    assert kind is AbortKind.CRASHED
    assert kind.invalidates


@pytest.mark.parametrize("probes", [1])
def test_the_probe_runs_once(probes: int) -> None:
    """A dead system is slow to answer; probing it in a loop would add minutes
    to a run that has already failed."""
    from theodb_bench.abort import describe_unavailability

    adapter = _Adapter(start_time=RUN_START + timedelta(seconds=30))

    describe_unavailability(adapter, run_started_at=RUN_START)

    assert adapter.probes == probes


def test_the_real_adapter_can_answer_the_probe() -> None:
    """The method has to exist on the adapter the runs actually use, or the
    diagnosis is dead code that reads as coverage."""
    from theodb_bench.adapters.postgres import PostgresAdapter

    assert callable(PostgresAdapter().postmaster_start_time)


def test_the_probe_reads_the_server_not_the_client_clock() -> None:
    """Comparing a server timestamp against a client clock would make the answer
    depend on drift between two machines. The SQL asks the server for both."""
    import inspect

    from theodb_bench.adapters.postgres import PostgresAdapter

    source = inspect.getsource(PostgresAdapter.postmaster_start_time)

    assert "pg_postmaster_start_time" in source


# ------------------------------------------- the diagnosis reaches the bundle
#
# A diagnosis that exists in a function nobody calls is worse than none: it reads
# as covered. The value is in `system.log`, which is what someone opens when a run
# comes back INVALID.


def test_the_bundle_records_whether_the_system_restarted(tmp_path: Any) -> None:
    from theodb_bench.bench.vector import VectorWorkload
    from theodb_bench.runner import RunRequest, run_benchmark

    workload = VectorWorkload(corpus_size=32, dimension=4, query_count=4, k=2)

    class _DyingAdapter(SystemAdapter):
        system_id = "theodb"

        def __init__(self) -> None:
            self.probed = False

        def capabilities(self) -> dict[str, bool]:
            return {"vector_exact": True}

        def prepare(self) -> None:
            return None

        def start(self) -> None:
            return None

        def wait_ready(self, timeout_seconds: float = 60.0) -> None:
            return None

        def load_dataset(self, spec: Any, vectors: Any) -> Any:
            raise SystemUnavailableError(
                "could not connect to theodb",
                context=ErrorContext(phase=Phase.DATASET_LOAD, system="theodb"),
            )

        def postmaster_start_time(self) -> Any:
            self.probed = True
            # Later than any plausible run start: a restart.
            return datetime.now(timezone.utc) + timedelta(hours=1)

        def build_index(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("never reached")

        def execute(self, query: Any) -> Any:
            raise AssertionError("never reached")

        def collect_stats(self) -> dict[str, Any]:
            return {}

        def export_config(self) -> dict[str, Any]:
            return {}

        def system_payload(self) -> dict[str, Any]:
            return {"system_id": "theodb"}

        def stop(self) -> None:
            return None

        def cleanup(self) -> None:
            return None

    outcome = run_benchmark(
        RunRequest(
            benchmark_id="vector/synthetic/smoke",
            workload=workload,
            adapter_factory=_DyingAdapter,
            results_root=tmp_path,
            repetitions=1,
            collect_process_telemetry=False,
        )
    )

    log = (outcome.bundle.raw_dir / "system.log").read_text()
    assert "restarted" in log, log
