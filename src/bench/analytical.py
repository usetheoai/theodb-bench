"""The analytical workload: the same data and the same queries, three ways.

Row store, columnar and Parquet execute identical queries over identical data.
That equality is the whole comparison: three paths measured on different data,
or answering different questions, would measure the data and the questions.

**Every query result is validated before its timing is accepted.** A path that
returns the wrong answer quickly is not a fast path, and this is the surface
where that failure is easiest to miss -- an aggregate is one number, and one
wrong number looks exactly like a right one.

Stage timings matter here more than anywhere else. A single wall time cannot
say whether a change moved metadata reading, row-group pruning, decode, filter
or aggregation, and "the query got faster" is not a finding until it does.

Naming: workloads derived from TPC-H are called **TPC-H-derived** and never
presented as an audited TPC result.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from theodb_bench.adapters.base import (
    AnalyticalQuery,
    AnalyticalResult,
    AnalyticalTable,
    SystemAdapter,
)
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.bench.vector import PointResult, RepetitionResult
from theodb_bench.errors import ConfigError, ErrorContext, Phase

ROW: Final[str] = "row"
COLUMNAR: Final[str] = "columnar"
PARQUET: Final[str] = "parquet"
PATHS: Final[tuple[str, ...]] = (ROW, COLUMNAR, PARQUET)

_CAPABILITY: Final[dict[str, str | None]] = {
    ROW: None,
    COLUMNAR: "columnar",
    PARQUET: "parquet",
}

COLUMNS: Final[tuple[str, ...]] = ("id", "amount", "category", "quantity")
CATEGORIES: Final[tuple[str, ...]] = ("a", "b", "c", "d")

QUERIES: Final[tuple[AnalyticalQuery, ...]] = (
    AnalyticalQuery(id="total_rows", description="Count every row."),
    AnalyticalQuery(id="sum_amount", description="Sum one column across the table."),
    AnalyticalQuery(
        id="group_by_category", description="Group and aggregate by a low-cardinality key."
    ),
    AnalyticalQuery(id="filtered_sum", description="Filter on two columns, then aggregate."),
)


@dataclass(frozen=True)
class AnalyticalWorkload:
    """A declarative analytical workload."""

    row_count: int
    seed: int = 20260813
    table: str = "bench_analytical"
    paths: tuple[str, ...] = PATHS
    queries: tuple[AnalyticalQuery, ...] = QUERIES
    repetitions: int = 3
    warmup_queries: int = 1

    def __post_init__(self) -> None:
        unknown = set(self.paths) - set(PATHS)
        if unknown:
            raise ConfigError(
                f"unknown execution path(s): {', '.join(sorted(unknown))}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.row_count < 1:
            raise ConfigError(
                "row_count must be at least 1", context=ErrorContext(phase=Phase.PREFLIGHT)
            )
        if self.repetitions < 1:
            raise ConfigError(
                "repetitions must be at least 1", context=ErrorContext(phase=Phase.PREFLIGHT)
            )

    def table_for(self, path: str) -> AnalyticalTable:
        return AnalyticalTable(name=f"{self.table}_{path}", columns=COLUMNS, path=path)

    # ----------------------------------------------------- Workload protocol

    def build(self, corpus: Any = None, queries: Any = None) -> AnalyticalBenchmark:
        """Rows come from this workload's own seed, so the arguments are ignored.

        A verified vector dataset has nothing to say about an analytical table,
        and taking one here would let a run declare a dataset identity it never
        measured.
        """
        return AnalyticalBenchmark(self)

    def benchmark_payload(self) -> dict[str, Any]:
        return {
            "workload": {
                "type": "analytical",
                "loop": "closed",
                # `k` is omitted rather than sent empty: the schema makes it
                # optional and requires it non-empty when present, which is the
                # right reading -- an aggregation has no k, and sending [] or [1]
                # would put a k-NN concept into a workload that has none.
                "operation_count": len(self.queries) * len(self.paths),
            },
            # An analytical answer is right or wrong against this benchmark's own
            # oracle. `exact_match` is the vocabulary's own term for that; a recall
            # figure would put a number where a verdict belongs.
            "quality": {"metric": "exact_match", "ground_truth": "computed"},
            "parameters": {"paths": list(self.paths)},
        }

    def expected_operations(self, measured_points: int, repetitions: int) -> int:
        return measured_points * repetitions

    @property
    def warmup_operations(self) -> int:
        return self.warmup_queries

    def quality_was_reported(self, points: list[Any]) -> bool:
        """Here the quality axis is the status, not a number.

        Every measured point had its answer compared to this benchmark's own
        oracle before it was called measured -- a point that disagreed carries
        `invalid` instead. Requiring a recall figure would invalidate a run for
        not producing a number that does not apply to it.
        """
        return any(point.status in {"measured", "invalid"} for point in points)


def generate_rows(workload: AnalyticalWorkload) -> list[tuple[Any, ...]]:
    """Seeded rows. Identical for every path, which is what makes them comparable."""
    rng = np.random.default_rng(workload.seed)
    amounts = rng.uniform(-100.0, 100.0, size=workload.row_count)
    categories = rng.choice(np.array(CATEGORIES), size=workload.row_count)
    quantities = rng.integers(1, 50, size=workload.row_count)
    return [
        (index, round(float(amounts[index]), 6), str(categories[index]), int(quantities[index]))
        for index in range(workload.row_count)
    ]


def expected_answer(rows: Sequence[tuple[Any, ...]], query_id: str) -> tuple[tuple[Any, ...], ...]:
    """The correct answer, computed here from the same rows.

    This is the oracle. Computing it in the benchmark rather than trusting any
    path is what makes a wrong answer detectable at all: if all three paths
    agreed on the same wrong number, comparing them to each other would find
    nothing.
    """
    if query_id == "total_rows":
        return ((len(rows),),)
    if query_id == "sum_amount":
        return ((round(sum(float(row[1]) for row in rows), 6),),)
    if query_id == "group_by_category":
        totals: dict[Any, float] = {}
        for row in rows:
            totals[row[2]] = totals.get(row[2], 0.0) + float(row[1])
        return tuple((key, round(value, 6)) for key, value in sorted(totals.items()))
    if query_id == "filtered_sum":
        selected = [row for row in rows if row[2] == "a" and float(row[1]) > 0]
        return ((round(sum(float(row[1]) for row in selected), 6),),)
    raise ConfigError(
        f"no oracle for query {query_id!r}", context=ErrorContext(phase=Phase.OFFLINE)
    )


def answers_match(
    observed: Sequence[tuple[Any, ...]],
    expected: Sequence[tuple[Any, ...]],
    tolerance: float,
) -> bool:
    """Compare answers, allowing float tolerance but not shape differences.

    Different row counts or different keys are a wrong answer, not a rounding
    difference, and no tolerance covers them.
    """
    if len(observed) != len(expected):
        return False
    for observed_row, expected_row in zip(observed, expected, strict=True):
        if len(observed_row) != len(expected_row):
            return False
        for left, right in zip(observed_row, expected_row, strict=True):
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if abs(float(left) - float(right)) > tolerance:
                    return False
            elif left != right:
                return False
    return True


@dataclass
class QueryMeasurement:
    """One query on one path."""

    query_id: str
    path: str
    status: str = "measured"
    status_detail: str | None = None
    latency: LatencySummary | None = None
    wall_seconds: float | None = None
    rows_processed: int | None = None
    bytes_read: int | None = None
    rows_per_second: float | None = None
    bytes_per_second: float | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query_id,
            "path": self.path,
            "status": self.status,
            "status_detail": self.status_detail,
            "wall_seconds": self.wall_seconds,
            "rows_processed": self.rows_processed,
            "bytes_read": self.bytes_read,
            "rows_per_second": self.rows_per_second,
            "bytes_per_second": self.bytes_per_second,
            "stage_seconds": dict(self.stage_seconds),
            "latency_ms": self.latency.as_dict() if self.latency else None,
        }

    def metric_series(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        if self.rows_per_second is not None:
            series["rows_per_second"] = [self.rows_per_second]
        if self.bytes_per_second is not None:
            series["bytes_per_second"] = [self.bytes_per_second]
        if self.latency is not None:
            for name in ("p50", "p95", "p99"):
                value = getattr(self.latency, name)
                if isinstance(value, float):
                    series[f"latency_{name}_ms"] = [value]
        for stage, seconds in self.stage_seconds.items():
            series[f"stage_{stage}_seconds"] = [seconds]
        return series


class AnalyticalBenchmark:
    """Runs the same queries over the same data on every declared path."""

    def __init__(self, workload: AnalyticalWorkload) -> None:
        self.workload = workload
        self.rows = generate_rows(workload)
        self.oracle = {query.id: expected_answer(self.rows, query.id) for query in workload.queries}

    def _load_path(self, adapter: SystemAdapter, path: str) -> float | None:
        """Load the identical rows into one path. Returns None when unsupported."""
        capability = _CAPABILITY[path]
        if capability is not None and not adapter.supports(capability):
            return None
        outcome = adapter.load_analytical(self.workload.table_for(path), self.rows)
        if not outcome.complete:
            raise ConfigError(
                f"{path}: loaded {outcome.rows_loaded} of {outcome.rows_expected} rows",
                context=ErrorContext(phase=Phase.DATASET_LOAD),
            )
        return outcome.seconds

    def run_query(
        self, adapter: SystemAdapter, path: str, query: AnalyticalQuery
    ) -> QueryMeasurement:
        """Validate first, then accept the timing."""
        measurement = QueryMeasurement(query_id=query.id, path=path)
        capability = _CAPABILITY[path]
        if capability is not None and not adapter.supports(capability):
            measurement.status = "unsupported"
            measurement.status_detail = f"{adapter.system_id} has no {path} path"
            return measurement

        table = self.workload.table_for(path)
        for _ in range(self.workload.warmup_queries):
            adapter.execute_analytical(table, query)

        latencies: list[float] = []
        last: AnalyticalResult | None = None
        for _ in range(self.workload.repetitions):
            outcome = adapter.execute_analytical(table, query)
            if not answers_match(outcome.rows, self.oracle[query.id], query.tolerance):
                # A wrong answer produced quickly is not a fast query, and its
                # timing is not evidence about anything.
                measurement.status = "invalid"
                measurement.status_detail = (
                    f"{path} returned {outcome.rows!r} for {query.id}; the oracle says "
                    f"{self.oracle[query.id]!r}. The timing was discarded."
                )
                return measurement
            latencies.append(outcome.wall_seconds * 1000.0)
            last = outcome

        measurement.latency = summarise_latency(latencies)
        measurement.wall_seconds = min(latencies) / 1000.0 if latencies else None
        if last is not None:
            measurement.rows_processed = last.rows_processed
            measurement.bytes_read = last.bytes_read
            measurement.stage_seconds = dict(last.stage_seconds)
            if measurement.wall_seconds and measurement.wall_seconds > 0:
                if last.rows_processed is not None:
                    measurement.rows_per_second = last.rows_processed / measurement.wall_seconds
                if last.bytes_read is not None:
                    measurement.bytes_per_second = last.bytes_read / measurement.wall_seconds
        return measurement

    def run(self, adapter: SystemAdapter) -> list[QueryMeasurement]:
        """Every declared query on every declared path."""
        measurements: list[QueryMeasurement] = []
        for path in self.workload.paths:
            if self._load_path(adapter, path) is None:
                measurements.extend(
                    QueryMeasurement(
                        query_id=query.id,
                        path=path,
                        status="unsupported",
                        status_detail=f"{adapter.system_id} has no {path} path",
                    )
                    for query in self.workload.queries
                )
                continue
            measurements.extend(
                self.run_query(adapter, path, query) for query in self.workload.queries
            )
        return measurements

    # --------------------------------------------------- Benchmark protocol

    def load(self, adapter: SystemAdapter, path: str | None = None) -> float | None:
        """Load one path, or every declared path when called by the orchestrator.

        The orchestrator loads once before measuring and does not know this
        family has three storage paths, so the no-argument form loads all of
        them. A path that the adapter does not support contributes None rather
        than zero: an absent path is not a path that loaded instantly.
        """
        if path is not None:
            return self._load_path(adapter, path)
        total = 0.0
        loaded_any = False
        for declared in self.workload.paths:
            seconds = self._load_path(adapter, declared)
            if seconds is not None:
                total += seconds
                loaded_any = True
        return total if loaded_any else None

    def points(self, adapter: SystemAdapter, repetitions: int) -> list[PointResult]:
        """Every query on every path, as one point each.

        The point is (query, path) rather than (query) because the paths are
        exactly what this workload compares: the same answer computed three ways
        is three operating points, and folding them into one would average away
        the comparison.
        """
        by_label: dict[str, PointResult] = {}
        for repetition in range(1, max(1, repetitions) + 1):
            for measurement in self.run(adapter):
                label = f"{measurement.query_id} via {measurement.path}"
                point = by_label.setdefault(
                    label,
                    PointResult(
                        label=label,
                        parameters={
                            "query": measurement.query_id,
                            "path": measurement.path,
                        },
                    ),
                )
                if measurement.status != "measured" or measurement.latency is None:
                    point.status = measurement.status
                    point.status_detail = measurement.status_detail
                    continue
                point.repetitions.append(
                    RepetitionResult(
                        repetition=repetition,
                        successes=1,
                        errors=0,
                        timeouts=0,
                        duration_seconds=measurement.wall_seconds or 0.0,
                        latency=measurement.latency,
                        # Correctness here is a right-or-wrong verdict, already
                        # carried by `status`. A recall figure would be a number
                        # standing in for a verdict.
                        recall=None,
                    )
                )
        return list(by_label.values())

    def compare_paths(self, measurements: Sequence[QueryMeasurement]) -> dict[str, Any]:
        """Line the paths up per query, so a difference is attributable.

        Only measured results are compared. An unsupported or invalid path
        appears with its status rather than being dropped, because a table that
        silently omits a path reads as though it was never tried.
        """
        by_query: dict[str, dict[str, Any]] = {}
        for measurement in measurements:
            entry = by_query.setdefault(measurement.query_id, {})
            entry[measurement.path] = {
                "status": measurement.status,
                "wall_seconds": measurement.wall_seconds,
                "rows_per_second": measurement.rows_per_second,
                "bytes_read": measurement.bytes_read,
                "stage_seconds": measurement.stage_seconds,
            }

        speedups: dict[str, dict[str, float]] = {}
        for query_id, paths in by_query.items():
            baseline = paths.get(ROW, {})
            base_seconds = baseline.get("wall_seconds")
            if not isinstance(base_seconds, float) or base_seconds <= 0:
                continue
            for path, entry in paths.items():
                seconds = entry.get("wall_seconds")
                if path == ROW or not isinstance(seconds, float) or seconds <= 0:
                    continue
                speedups.setdefault(query_id, {})[path] = base_seconds / seconds

        return {
            "queries": by_query,
            "speedup_over_row": speedups,
            "note": (
                "The same rows and the same queries on every path. A speedup here "
                "describes the execution path, not the data or the question."
            ),
        }


def timed_reference_scan(
    rows: Sequence[tuple[Any, ...]], query_id: str
) -> tuple[float, tuple[tuple[Any, ...], ...]]:
    """Compute the answer in process, as a floor to read a path against.

    Not a competitor: an in-memory Python loop with no storage, durability or
    concurrency. The report says exactly that, so nobody reads it as one.
    """
    started = time.perf_counter()
    answer = expected_answer(rows, query_id)
    return time.perf_counter() - started, answer
