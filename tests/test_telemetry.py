"""A collector that cannot measure must say so, not report zero."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from theodb_bench.absent import AbsenceReason, Absent
from theodb_bench.telemetry import (
    CollectorSet,
    PerfStatCollector,
    ProcessCollector,
    sample_process,
)

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only")


# ------------------------------------------------------------------- process


@linux_only
def test_process_sample_reads_real_counters() -> None:
    sample = sample_process(os.getpid())
    assert sample.rss_bytes is not None and sample.rss_bytes > 0
    assert sample.cpu_seconds is not None and sample.cpu_seconds >= 0


def test_sampling_a_missing_process_yields_no_values() -> None:
    sample = sample_process(2**31 - 1)
    assert sample.rss_bytes is None
    assert sample.cpu_seconds is None


@linux_only
def test_process_collector_measures_work_it_observed() -> None:
    collector = ProcessCollector(os.getpid())
    collector.start()
    # Do enough arithmetic that the CPU delta is not rounding noise.
    total = sum(i * i for i in range(400_000))
    assert total > 0
    collector.stop()
    metrics = collector.result()
    assert not isinstance(metrics["cpu_seconds"], Absent)
    assert metrics["cpu_seconds"] >= 0
    assert not isinstance(metrics["rss_bytes"], Absent)


@linux_only
def test_process_collector_tracks_peak_rss_across_samples() -> None:
    collector = ProcessCollector(os.getpid())
    collector.start()
    ballast = bytearray(24 * 1024 * 1024)
    collector.sample()
    del ballast
    collector.stop()
    peak = collector.result()["peak_rss_bytes"]
    current = collector.result()["rss_bytes"]
    assert not isinstance(peak, Absent)
    assert not isinstance(current, Absent)
    assert peak >= current


def test_collector_for_a_dead_process_reports_absence_not_zero() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    collector = ProcessCollector(child.pid)
    collector.start()
    collector.stop()
    for name, value in collector.result().items():
        assert isinstance(value, Absent), f"{name} fabricated a value for a dead process"


def test_a_disabled_collector_reports_not_collected() -> None:
    collector = ProcessCollector(os.getpid(), enabled=False)
    collector.start()
    collector.stop()
    for value in collector.result().values():
        assert isinstance(value, Absent)
        assert value.reason is AbsenceReason.NOT_COLLECTED


def test_a_collector_that_never_started_is_unavailable_not_zero() -> None:
    collector = ProcessCollector(os.getpid())
    for value in collector.result().values():
        assert isinstance(value, Absent)
        assert value.reason is AbsenceReason.UNAVAILABLE


# ---------------------------------------------------------------------- perf


def test_perf_collector_never_fabricates_counters() -> None:
    # On a host where perf_event_paranoid denies access this must produce
    # absences with the reason, and on a permissive host real numbers. Either
    # is acceptable; a zero is not.
    collector = PerfStatCollector(os.getpid())
    collector.start()
    sum(i for i in range(50_000))
    collector.stop()
    for name, value in collector.result().items():
        if isinstance(value, Absent):
            assert value.detail, f"{name} absent without a reason"
        else:
            assert value >= 0


def test_perf_parses_unsupported_counters_as_absences() -> None:
    collector = PerfStatCollector(os.getpid())
    parsed = collector._parse(
        "<not supported>,,cycles\n1234,,instructions\n<not counted>,,cache-misses\n"
    )
    assert isinstance(parsed["cycles"], Absent)
    assert parsed["instructions"] == 1234.0
    assert isinstance(parsed["cache_misses"], Absent)


def test_perf_parses_a_real_csv_row() -> None:
    collector = PerfStatCollector(os.getpid())
    parsed = collector._parse("9876543,,cycles,1000000,100.00,,")
    assert parsed["cycles"] == 9876543.0


def test_disabled_perf_collector_costs_nothing_and_says_so() -> None:
    collector = PerfStatCollector(os.getpid(), enabled=False)
    collector.start()
    collector.stop()
    values = collector.result()
    assert set(values) == {e.replace("-", "_") for e in collector.events}
    assert all(isinstance(v, Absent) for v in values.values())


# ------------------------------------------------------------- collector set


@linux_only
def test_collector_set_namespaces_metrics_by_collector() -> None:
    collectors = CollectorSet().add(ProcessCollector(os.getpid()))
    collectors.start()
    collectors.stop()
    assert all(key.startswith("process.") for key in collectors.results())


def test_collector_set_measures_its_own_overhead() -> None:
    # A runner that cannot say what its instrumentation cost cannot claim the
    # instrumentation was free.
    collectors = CollectorSet().add(ProcessCollector(os.getpid()))
    collectors.start()
    collectors.sample()
    collectors.stop()
    collectors.results()
    assert collectors.overhead_seconds > 0


def test_empty_collector_set_is_valid_and_reports_nothing() -> None:
    collectors = CollectorSet()
    collectors.start()
    collectors.stop()
    assert collectors.results() == {}


def test_collector_set_serialises_which_collectors_ran() -> None:
    collectors = CollectorSet()
    collectors.add(ProcessCollector(os.getpid()))
    collectors.add(PerfStatCollector(os.getpid(), enabled=False))
    collectors.start()
    collectors.stop()
    payload = collectors.as_dict()
    assert payload["collectors"] == ["process", "perf"]
    assert payload["enabled"] == ["process"]
    assert "overhead_seconds" in payload
