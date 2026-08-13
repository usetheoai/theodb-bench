"""The operations workload: two clocks that must never be conflated.

**The foreground clock** is what the writing transaction experienced. Moving
embedding work into a background worker improves it, and a benchmark that
stopped there would report an unambiguous win.

**The freshness clock** is how long until the derived embedding is queryable.
It is the cost the foreground clock stopped paying. A system can make writes
arbitrarily fast by deferring more work, and the only thing that reveals the
trade is measuring when the data actually became usable.

Reporting either alone is a way of describing the design as free. This module
measures both and reports them side by side.

For the agent surface, the freshness clock is `read-your-writes`: an
observation written at step N must be retrievable at step N+1. There, exceeding
the declared bound is a correctness failure rather than a slow number
(`docs/methodology/AGENT-WORKLOAD.md` §4).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.adapters.base import Document, DocumentTableSpec, SystemAdapter
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.errors import ConfigError, ErrorContext, Phase

DEFAULT_TABLE: Final[str] = "bench_operations"

INSERT_BASELINE: Final[str] = "insert_baseline"
INSERT_VECTORIZED: Final[str] = "insert_vectorized"
UPDATE_SOURCE: Final[str] = "update_source"
BACKLOG_DRAIN: Final[str] = "backlog_drain"
WORKER_SATURATION: Final[str] = "worker_saturation"

WORKLOADS: Final[tuple[str, ...]] = (
    INSERT_BASELINE,
    INSERT_VECTORIZED,
    UPDATE_SOURCE,
    BACKLOG_DRAIN,
    WORKER_SATURATION,
)


@dataclass(frozen=True)
class OperationsWorkload:
    """A declarative operations workload."""

    row_count: int
    dimension: int = 32
    seed: int = 20260813
    table: str = DEFAULT_TABLE
    workloads: tuple[str, ...] = WORKLOADS
    freshness_timeout_seconds: float = 30.0
    """How long to wait for the second clock before declaring the row stale.

    A bound, not a convenience: "eventually retrievable" is not a bound, and a
    measurement that waited indefinitely would never report a failure."""

    freshness_poll_seconds: float = 0.002
    saturation_write_rate: float = 500.0
    """Foreground writes per second to push during the saturation workload."""

    def __post_init__(self) -> None:
        unknown = set(self.workloads) - set(WORKLOADS)
        if unknown:
            raise ConfigError(
                f"unknown operations workload(s): {', '.join(sorted(unknown))}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.row_count < 1:
            raise ConfigError(
                "row_count must be at least 1", context=ErrorContext(phase=Phase.PREFLIGHT)
            )

    def table_spec(self) -> DocumentTableSpec:
        return DocumentTableSpec(table=self.table, dimension=self.dimension)


def generate_rows(workload: OperationsWorkload, offset: int = 0) -> list[Document]:
    """Seeded rows to write."""
    rng = np.random.default_rng(workload.seed + offset)
    rows: list[Document] = []
    for index in range(workload.row_count):
        vector: npt.NDArray[np.float32] = rng.standard_normal(workload.dimension).astype(np.float32)
        rows.append(
            Document(id=offset + index, text=f"row {offset + index} content token", vector=vector)
        )
    return rows


@dataclass
class OperationsResult:
    """One operations workload, with both clocks kept apart."""

    workload: str
    status: str = "measured"
    status_detail: str | None = None

    write_latency: LatencySummary | None = None
    write_throughput: float | None = None
    writes: int = 0

    freshness_latency: LatencySummary | None = None
    stale_rows: int = 0
    """Rows that never became fresh within the declared bound. Counted, not
    averaged into the freshness distribution."""

    peak_queue_depth: int | None = None
    worker_throughput: float | None = None
    retries: int | None = None
    failures: int | None = None

    def metric_series(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        if self.write_throughput is not None:
            series["write_throughput_per_second"] = [self.write_throughput]
        if self.worker_throughput is not None:
            series["worker_throughput_per_second"] = [self.worker_throughput]
        for prefix, summary in (
            ("write_latency", self.write_latency),
            ("freshness", self.freshness_latency),
        ):
            if summary is None:
                continue
            for name in ("p50", "p95", "p99"):
                value = getattr(summary, name)
                if isinstance(value, float):
                    series[f"{prefix}_{name}_ms"] = [value]
        if self.peak_queue_depth is not None:
            series["peak_queue_depth"] = [float(self.peak_queue_depth)]
        return series

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "status": self.status,
            "status_detail": self.status_detail,
            "writes": self.writes,
            "write_throughput_per_second": self.write_throughput,
            "write_latency_ms": self.write_latency.as_dict() if self.write_latency else None,
            "freshness_ms": (self.freshness_latency.as_dict() if self.freshness_latency else None),
            "stale_rows": self.stale_rows,
            "peak_queue_depth": self.peak_queue_depth,
            "worker_throughput_per_second": self.worker_throughput,
            "retries": self.retries,
            "failures": self.failures,
        }


class OperationsBenchmark:
    """Measures the foreground clock and the freshness clock separately."""

    def __init__(self, workload: OperationsWorkload) -> None:
        self.workload = workload

    def run(self, adapter: SystemAdapter, name: str) -> OperationsResult:
        if name not in WORKLOADS:
            raise ConfigError(
                f"unknown operations workload {name!r}",
                context=ErrorContext(phase=Phase.MEASUREMENT),
            )
        if not adapter.supports("vectorizer"):
            return OperationsResult(
                workload=name,
                status="unsupported",
                status_detail=f"{adapter.system_id} has no background vectorizer",
            )

        if name == INSERT_BASELINE:
            return self._insert(adapter, measure_freshness=False)
        if name == INSERT_VECTORIZED:
            return self._insert(adapter, measure_freshness=True)
        if name == UPDATE_SOURCE:
            return self._update(adapter)
        if name == BACKLOG_DRAIN:
            return self._backlog(adapter)
        return self._saturation(adapter)

    # ---------------------------------------------------------------- inserts

    def _insert(self, adapter: SystemAdapter, *, measure_freshness: bool) -> OperationsResult:
        spec = self.workload.table_spec()
        rows = generate_rows(self.workload)
        result = OperationsResult(
            workload=INSERT_VECTORIZED if measure_freshness else INSERT_BASELINE
        )

        write_latencies: list[float] = []
        written_at: dict[int, float] = {}
        started = time.perf_counter()
        for row in rows:
            outcome = adapter.insert_document(spec, row)
            write_latencies.append(outcome.latency_seconds * 1000.0)
            written_at[row.id] = time.perf_counter()
        duration = time.perf_counter() - started

        result.writes = len(write_latencies)
        result.write_latency = summarise_latency(write_latencies)
        result.write_throughput = len(write_latencies) / duration if duration > 0 else None

        if measure_freshness:
            freshness, stale = self._await_freshness(adapter, spec, written_at)
            result.freshness_latency = summarise_latency(freshness)
            result.stale_rows = stale
            self._record_worker(adapter, result, duration)
        return result

    def _update(self, adapter: SystemAdapter) -> OperationsResult:
        """Rewrite a source column and time how long the derived value lags.

        Harder than an insert: the row already has an embedding, and a system
        that does not invalidate it would look instantly fresh while serving a
        stale answer.
        """
        spec = self.workload.table_spec()
        rows = generate_rows(self.workload)
        for row in rows:
            adapter.insert_document(spec, row)
        self._await_freshness(adapter, spec, {row.id: time.perf_counter() for row in rows})

        result = OperationsResult(workload=UPDATE_SOURCE)
        write_latencies: list[float] = []
        written_at: dict[int, float] = {}
        started = time.perf_counter()
        for row in rows:
            outcome = adapter.update_document_text(spec, row.id, f"{row.text} revised")
            write_latencies.append(outcome.latency_seconds * 1000.0)
            written_at[row.id] = time.perf_counter()
        duration = time.perf_counter() - started

        result.writes = len(write_latencies)
        result.write_latency = summarise_latency(write_latencies)
        result.write_throughput = len(write_latencies) / duration if duration > 0 else None
        freshness, stale = self._await_freshness(adapter, spec, written_at)
        result.freshness_latency = summarise_latency(freshness)
        result.stale_rows = stale
        self._record_worker(adapter, result, duration)
        return result

    # ---------------------------------------------------------------- backlog

    def _backlog(self, adapter: SystemAdapter) -> OperationsResult:
        """Write everything as fast as possible, then time the drain."""
        spec = self.workload.table_spec()
        rows = generate_rows(self.workload)
        result = OperationsResult(workload=BACKLOG_DRAIN)

        write_latencies: list[float] = []
        started = time.perf_counter()
        for row in rows:
            outcome = adapter.insert_document(spec, row)
            write_latencies.append(outcome.latency_seconds * 1000.0)
        write_duration = time.perf_counter() - started

        result.writes = len(write_latencies)
        result.write_latency = summarise_latency(write_latencies)
        result.write_throughput = (
            len(write_latencies) / write_duration if write_duration > 0 else None
        )
        result.peak_queue_depth = adapter.queue_depth()

        drain_started = time.perf_counter()
        deadline = drain_started + self.workload.freshness_timeout_seconds
        while adapter.queue_depth() > 0 and time.perf_counter() < deadline:
            time.sleep(self.workload.freshness_poll_seconds)
        drain_seconds = time.perf_counter() - drain_started

        remaining = adapter.queue_depth()
        if remaining:
            result.stale_rows = remaining
            result.status_detail = (
                f"{remaining} row(s) still queued after {self.workload.freshness_timeout_seconds}s"
            )
        result.worker_throughput = (
            (len(rows) - remaining) / drain_seconds if drain_seconds > 0 else None
        )
        self._record_worker(adapter, result, drain_seconds)
        return result

    # ------------------------------------------------------------- saturation

    def _saturation(self, adapter: SystemAdapter) -> OperationsResult:
        """Push writes faster than the worker can absorb them.

        The finding is not a latency number: it is whether the queue grows
        without bound. A system whose backlog grows monotonically under a
        sustained write rate has a capacity limit, and that is the result.
        """
        spec = self.workload.table_spec()
        rows = generate_rows(self.workload)
        result = OperationsResult(workload=WORKER_SATURATION)

        interval = 1.0 / self.workload.saturation_write_rate
        depths: list[int] = []
        write_latencies: list[float] = []
        started = time.perf_counter()
        for index, row in enumerate(rows):
            outcome = adapter.insert_document(spec, row)
            write_latencies.append(outcome.latency_seconds * 1000.0)
            if index % 10 == 0:
                depths.append(adapter.queue_depth())
            target = started + (index + 1) * interval
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        duration = time.perf_counter() - started

        result.writes = len(write_latencies)
        result.write_latency = summarise_latency(write_latencies)
        result.write_throughput = len(write_latencies) / duration if duration > 0 else None
        result.peak_queue_depth = max(depths) if depths else adapter.queue_depth()
        # Growing backlog is the observation; the report says so rather than
        # burying it in a percentile.
        if len(depths) >= 2 and depths[-1] > depths[0]:
            result.status_detail = (
                f"backlog grew from {depths[0]} to {depths[-1]} under a sustained "
                f"{self.workload.saturation_write_rate:.0f} writes/s"
            )
        self._record_worker(adapter, result, duration)
        return result

    # ------------------------------------------------------------------ shared

    def _await_freshness(
        self, adapter: SystemAdapter, spec: DocumentTableSpec, written_at: dict[int, float]
    ) -> tuple[list[float], int]:
        """Time each row from its write until it becomes queryable.

        A row that never becomes fresh within the bound is **counted**, not
        given a latency. Averaging in a timeout would let a system that lost
        the row look merely slow.
        """
        pending = dict(written_at)
        freshness: list[float] = []
        deadline = time.perf_counter() + self.workload.freshness_timeout_seconds

        while pending and time.perf_counter() < deadline:
            for row_id in list(pending):
                if adapter.is_fresh(spec, row_id):
                    freshness.append((time.perf_counter() - pending.pop(row_id)) * 1000.0)
            if pending:
                time.sleep(self.workload.freshness_poll_seconds)
        return freshness, len(pending)

    def _record_worker(
        self, adapter: SystemAdapter, result: OperationsResult, duration: float
    ) -> None:
        stats = adapter.vectorizer_stats()
        result.retries = int(stats.get("retries", 0))
        result.failures = int(stats.get("failures", 0))
        if result.worker_throughput is None and duration > 0:
            processed = int(stats.get("processed", 0))
            result.worker_throughput = processed / duration
        if result.peak_queue_depth is None:
            result.peak_queue_depth = int(stats.get("queue_depth", 0))


def compare_clocks(results: Sequence[OperationsResult]) -> dict[str, Any]:
    """Put the two clocks side by side.

    The comparison a vectorizer benchmark exists to make: the foreground gets
    faster, and the question is what that cost on the other clock.
    """
    baseline = next((r for r in results if r.workload == INSERT_BASELINE), None)
    vectorized = next((r for r in results if r.workload == INSERT_VECTORIZED), None)

    payload: dict[str, Any] = {
        "workloads": {result.workload: result.as_dict() for result in results}
    }
    if baseline is None or vectorized is None:
        payload["foreground_delta"] = None
        payload["note"] = (
            "Both the baseline and the vectorized insert are needed to state what "
            "deferring embedding work cost."
        )
        return payload

    base_p50 = baseline.write_latency.p50 if baseline.write_latency else None
    vec_p50 = vectorized.write_latency.p50 if vectorized.write_latency else None
    fresh_p50 = vectorized.freshness_latency.p50 if vectorized.freshness_latency else None

    payload["foreground_delta_ms"] = (
        vec_p50 - base_p50 if isinstance(base_p50, float) and isinstance(vec_p50, float) else None
    )
    payload["freshness_p50_ms"] = fresh_p50 if isinstance(fresh_p50, float) else None
    payload["note"] = (
        "The foreground clock and the freshness clock describe the same design "
        "from two sides. Reporting either alone presents deferred work as free."
    )
    return payload
