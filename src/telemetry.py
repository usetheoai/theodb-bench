"""Telemetry collectors.

Three rules shape this module.

A collector that could not measure something reports an absence carrying the
reason, never a zero -- zero cache misses is a finding, and an unavailable
counter is not.

A collector may be switched off. Telemetry that cannot be disabled cannot be
excluded from the measurement, and the runner is not allowed to be the thing
that changes the number (TRD section 37).

A collector's own cost is measurable. ``CollectorSet`` accumulates the wall
time it spends inside collectors, so a run can report what its instrumentation
cost instead of assuming it was free.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from theodb_bench.absent import Absent, Measured, encode, not_collected, unavailable
from theodb_bench.command import which

_PROC: Final[Path] = Path("/proc")
_CLOCK_TICKS: Final[int] = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


class Collector(ABC):
    """One independent source of telemetry."""

    name: str

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._started = False

    @abstractmethod
    def _start(self) -> None: ...

    @abstractmethod
    def _stop(self) -> None: ...

    @abstractmethod
    def _result(self) -> dict[str, Measured[float]]:
        """Metrics gathered between start and stop."""

    def start(self) -> None:
        if not self.enabled:
            return
        self._start()
        self._started = True

    def stop(self) -> None:
        if not self.enabled or not self._started:
            return
        self._stop()

    def result(self) -> dict[str, Measured[float]]:
        if not self.enabled:
            return {
                key: not_collected(f"{self.name} collector disabled for this run")
                for key in self.metric_names()
            }
        if not self._started:
            return {
                key: unavailable(f"{self.name} collector never started")
                for key in self.metric_names()
            }
        return self._result()

    @abstractmethod
    def metric_names(self) -> tuple[str, ...]:
        """Every metric this collector can produce, present or absent."""


# ------------------------------------------------------------------- process


def _read_proc_status(pid: int) -> dict[str, str]:
    try:
        raw = (_PROC / str(pid) / "status").read_text(encoding="utf-8")
    except OSError:
        return {}
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _read_proc_stat(pid: int) -> list[str] | None:
    try:
        raw = (_PROC / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # The command field may contain spaces and is parenthesised; everything
    # after it is positional, so split on the last ')'.
    _, _, rest = raw.partition(") ")
    return rest.split()


def _kb_field(fields: dict[str, str], key: str) -> int | None:
    raw = fields.get(key)
    if raw is None:
        return None
    parts = raw.split()
    try:
        return int(parts[0]) * 1024
    except (ValueError, IndexError):
        return None


@dataclass
class _ProcessSample:
    rss_bytes: int | None
    peak_rss_bytes: int | None
    cpu_seconds: float | None
    voluntary_switches: int | None
    involuntary_switches: int | None
    minor_faults: int | None
    major_faults: int | None


def sample_process(pid: int) -> _ProcessSample:
    status = _read_proc_status(pid)
    stat = _read_proc_stat(pid)

    cpu_seconds: float | None = None
    minor_faults: int | None = None
    major_faults: int | None = None
    if stat is not None and len(stat) >= 15:
        try:
            # Fields (0-based after the command): 7 minflt, 9 majflt,
            # 11 utime, 12 stime.
            minor_faults = int(stat[7])
            major_faults = int(stat[9])
            cpu_seconds = (int(stat[11]) + int(stat[12])) / _CLOCK_TICKS
        except (ValueError, IndexError):
            cpu_seconds = None

    def as_int(key: str) -> int | None:
        raw = status.get(key)
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None

    return _ProcessSample(
        rss_bytes=_kb_field(status, "VmRSS"),
        peak_rss_bytes=_kb_field(status, "VmHWM"),
        cpu_seconds=cpu_seconds,
        voluntary_switches=as_int("voluntary_ctxt_switches"),
        involuntary_switches=as_int("nonvoluntary_ctxt_switches"),
        minor_faults=minor_faults,
        major_faults=major_faults,
    )


class ProcessCollector(Collector):
    """Per-process resource usage read from procfs.

    Cheap enough to sample during a measurement window: two small pseudo-file
    reads with no allocation beyond the parsed strings.
    """

    name = "process"

    def __init__(self, pid: int, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.pid = pid
        self._first: _ProcessSample | None = None
        self._last: _ProcessSample | None = None
        self._peak_rss: int | None = None

    def metric_names(self) -> tuple[str, ...]:
        return (
            "rss_bytes",
            "peak_rss_bytes",
            "cpu_seconds",
            "context_switches",
            "minor_faults",
            "major_faults",
        )

    def _start(self) -> None:
        self._first = sample_process(self.pid)
        self._last = self._first
        self._peak_rss = self._first.rss_bytes

    def sample(self) -> None:
        """Take an intermediate sample. Safe to call from a monitoring loop."""
        if not self.enabled or not self._started:
            return
        current = sample_process(self.pid)
        self._last = current
        if current.rss_bytes is not None:
            self._peak_rss = max(self._peak_rss or 0, current.rss_bytes)

    def _stop(self) -> None:
        self.sample()

    def _result(self) -> dict[str, Measured[float]]:
        first, last = self._first, self._last
        if first is None or last is None:
            return {key: unavailable("no process sample taken") for key in self.metric_names()}

        def delta(attr: str) -> Measured[float]:
            start_value: float | None = getattr(first, attr)
            end_value: float | None = getattr(last, attr)
            if start_value is None or end_value is None:
                return unavailable(f"/proc/{self.pid} did not expose {attr}")
            return float(end_value - start_value)

        switches: Measured[float]
        voluntary = delta("voluntary_switches")
        involuntary = delta("involuntary_switches")
        if isinstance(voluntary, Absent) or isinstance(involuntary, Absent):
            switches = unavailable(f"/proc/{self.pid} did not expose context switch counters")
        else:
            switches = voluntary + involuntary

        peak: Measured[float] = (
            self._peak_rss
            if self._peak_rss is not None
            else unavailable(f"/proc/{self.pid} did not expose VmRSS")
        )
        rss: Measured[float] = (
            last.rss_bytes
            if last.rss_bytes is not None
            else unavailable(f"/proc/{self.pid} did not expose VmRSS")
        )

        return {
            "rss_bytes": rss,
            "peak_rss_bytes": peak,
            "cpu_seconds": delta("cpu_seconds"),
            "context_switches": switches,
            "minor_faults": delta("minor_faults"),
            "major_faults": delta("major_faults"),
        }


# ---------------------------------------------------------------------- perf


PERF_EVENTS: Final[tuple[str, ...]] = (
    "cycles",
    "instructions",
    "cache-misses",
    "cache-references",
    "branch-misses",
)


class PerfStatCollector(Collector):
    """Hardware counters via ``perf stat``, attached to a running process.

    Every failure mode here ends in an absence with a reason. A host with
    ``perf_event_paranoid`` set to 3 is common, and a benchmark that reported
    zero cycles there would be lying in a way nobody would notice.
    """

    name = "perf"

    def __init__(
        self,
        pid: int,
        *,
        events: tuple[str, ...] = PERF_EVENTS,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.pid = pid
        self.events = events
        self._process: subprocess.Popen[str] | None = None
        self._reason: str | None = None
        self._parsed: dict[str, Measured[float]] = {}

    def metric_names(self) -> tuple[str, ...]:
        return tuple(event.replace("-", "_") for event in self.events)

    def _start(self) -> None:
        if which("perf") is None:
            self._reason = "perf not on PATH"
            return
        paranoid_path = _PROC / "sys/kernel/perf_event_paranoid"
        try:
            paranoid = int(paranoid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            paranoid = None
        if paranoid is not None and paranoid > 2:
            self._reason = f"perf_event_paranoid={paranoid} denies per-process counters"
            return
        argv = [
            "perf",
            "stat",
            "-x",
            ",",
            "-e",
            ",".join(self.events),
            "-p",
            str(self.pid),
        ]
        try:
            self._process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            self._reason = f"perf could not be started: {exc}"

    def _stop(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            self._reason = "perf did not terminate; counters discarded"
            return
        except OSError as exc:
            self._reason = f"perf could not be stopped: {exc}"
            return
        self._parsed = self._parse(stderr)

    def _parse(self, output: str) -> dict[str, Measured[float]]:
        """Parse ``perf stat -x,`` CSV.

        A counter perf itself marks unsupported or not-counted becomes an
        absence, not a zero.
        """
        parsed: dict[str, Measured[float]] = {}
        for line in output.splitlines():
            parts = line.split(",")
            if len(parts) < 3:
                continue
            raw_value, _unit, event = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not event:
                continue
            key = event.replace("-", "_")
            if raw_value in {"<not supported>", "<not counted>", ""}:
                parsed[key] = unavailable(f"perf reported {raw_value or 'no value'} for {event}")
                continue
            try:
                parsed[key] = float(raw_value)
            except ValueError:
                parsed[key] = unavailable(f"unparseable perf value {raw_value!r} for {event}")
        return parsed

    def _result(self) -> dict[str, Measured[float]]:
        if self._reason is not None:
            return {key: unavailable(self._reason) for key in self.metric_names()}
        if not self._parsed:
            return {
                key: unavailable("perf produced no counter output") for key in self.metric_names()
            }
        return {
            key: self._parsed.get(key, unavailable(f"perf did not report {key}"))
            for key in self.metric_names()
        }


# ---------------------------------------------------------------- collector set


@dataclass
class CollectorSet:
    """A run's collectors, plus the cost of running them."""

    collectors: list[Collector] = field(default_factory=list)
    overhead_seconds: float = 0.0

    def add(self, collector: Collector) -> CollectorSet:
        self.collectors.append(collector)
        return self

    def _timed(self, action: str) -> None:
        started = time.perf_counter()
        for collector in self.collectors:
            getattr(collector, action)()
        self.overhead_seconds += time.perf_counter() - started

    def start(self) -> None:
        self._timed("start")

    def stop(self) -> None:
        self._timed("stop")

    def sample(self) -> None:
        """Sample every collector that supports intermediate sampling."""
        started = time.perf_counter()
        for collector in self.collectors:
            sampler = getattr(collector, "sample", None)
            if callable(sampler):
                sampler()
        self.overhead_seconds += time.perf_counter() - started

    def results(self) -> dict[str, Measured[float]]:
        """Merged metrics, namespaced by collector so two cannot collide."""
        merged: dict[str, Measured[float]] = {}
        started = time.perf_counter()
        for collector in self.collectors:
            for key, value in collector.result().items():
                merged[f"{collector.name}.{key}"] = value
        self.overhead_seconds += time.perf_counter() - started
        return merged

    def as_dict(self) -> dict[str, Any]:
        return {
            "collectors": [c.name for c in self.collectors],
            "enabled": [c.name for c in self.collectors if c.enabled],
            "overhead_seconds": round(self.overhead_seconds, 6),
            "metrics": {key: encode(value) for key, value in self.results().items()},
        }
