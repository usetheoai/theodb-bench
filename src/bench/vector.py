"""The vector ANN workload.

Defines what is measured and nothing about how a system starts: every system
interaction goes through the adapter contract (TRD D2).

The measurement rules that matter here, each of them an invariant rather than a
preference (``docs/methodology/MEASUREMENT-INTEGRITY.md``):

Warm-up is untimed and separate, so percentiles describe a consistently warm
cache instead of a mixture (I7).

Indexes belonging to other configurations are dropped before a point is
measured, because two indexes of the same family on the same column let the
system choose and one sweep flattens onto the other (I6).

A reduced query sample appears in the point label, never silently (I9).

Failed and timed-out queries are counted apart from successes and never enter
the latency distribution (objective §23).
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.adapters.base import (
    IndexSpec,
    KnnQuery,
    SystemAdapter,
    VectorTableSpec,
)
from theodb_bench.analysis.quality import brute_force_ground_truth, recall_at_k
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.errors import (
    BenchError,
    ConfigError,
    ErrorContext,
    MeasurementError,
    Phase,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)

DEFAULT_TABLE: Final[str] = "bench_vectors"

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class VectorWorkload:
    """A declarative ANN workload."""

    corpus_size: int
    dimension: int
    query_count: int
    k: int
    metric: str = "l2"
    seed: int = 20260813
    table: str = DEFAULT_TABLE
    indexes: tuple[IndexSpec, ...] = (IndexSpec(kind="none"),)
    search_sweep: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    warmup_queries: int = 0
    query_cap: int | None = None
    """Reduce the measured sample for an index that is O(N) per query. The cap
    lands in the point label; it is never applied silently."""

    def __post_init__(self) -> None:
        if self.k > self.corpus_size:
            raise ConfigError(
                f"k={self.k} exceeds the corpus size {self.corpus_size}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.query_count < 1 or self.corpus_size < 1:
            raise ConfigError(
                "corpus and query set must both be non-empty",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )

    def sweep_points(self) -> list[dict[str, Any]]:
        """Every combination of the declared search parameters."""
        if not self.search_sweep:
            return [{}]
        names = sorted(self.search_sweep)
        return [
            dict(zip(names, combination, strict=True))
            for combination in itertools.product(*(self.search_sweep[name] for name in names))
        ]

    def table_spec(self) -> VectorTableSpec:
        return VectorTableSpec(table=self.table, dimension=self.dimension, metric=self.metric)


def generate_corpus(workload: VectorWorkload) -> tuple[FloatArray, FloatArray]:
    """A seeded corpus and query set.

    Seeded on purpose: the same seed must give bit-identical data, so that two
    runs of the same benchmark differ only in the system under test. Public
    analogues do not seed, and their numbers cannot be reproduced exactly.
    """
    rng = np.random.default_rng(workload.seed)
    corpus = rng.standard_normal((workload.corpus_size, workload.dimension), dtype=np.float32)
    queries = rng.standard_normal((workload.query_count, workload.dimension), dtype=np.float32)
    return corpus, queries


@dataclass
class RepetitionResult:
    """One repetition of one configuration."""

    repetition: int
    successes: int
    errors: int
    timeouts: int
    duration_seconds: float
    latency: LatencySummary
    recall: float | None
    build_seconds: float | None = None
    index_size_bytes: int | None = None

    @property
    def throughput(self) -> float | None:
        return self.successes / self.duration_seconds if self.duration_seconds > 0 else None


@dataclass
class PointResult:
    """One benchmark configuration across its repetitions."""

    label: str
    parameters: dict[str, Any]
    status: str = "measured"
    status_detail: str | None = None
    repetitions: list[RepetitionResult] = field(default_factory=list)

    def metric_series(self) -> dict[str, list[float]]:
        """Per-metric values across repetitions, for aggregation."""
        series: dict[str, list[float]] = {}
        for repetition in self.repetitions:
            throughput = repetition.throughput
            if throughput is not None:
                series.setdefault("throughput_per_second", []).append(throughput)
            if repetition.recall is not None:
                series.setdefault("recall", []).append(repetition.recall)
            for name in ("p50", "p95", "p99"):
                value = getattr(repetition.latency, name)
                if isinstance(value, float):
                    series.setdefault(f"latency_{name}_ms", []).append(value)
            if repetition.build_seconds is not None:
                series.setdefault("build_seconds", []).append(repetition.build_seconds)
            if repetition.index_size_bytes is not None:
                series.setdefault("index_size_bytes", []).append(float(repetition.index_size_bytes))
        return series


def build_label(index: IndexSpec, search: dict[str, Any], query_cap: int | None) -> str:
    """A label a reader can map back to a configuration.

    A reduced query sample is part of the label. Hiding it would let two points
    measured over different sample sizes sit in the same table as equals.
    """
    parts = [index.label()]
    if search:
        parts.append(" ".join(f"{key}={value}" for key, value in sorted(search.items())))
    if query_cap is not None:
        parts.append(f"[q={query_cap}]")
    return " ".join(parts)


def _check_supplied(workload: VectorWorkload, corpus: FloatArray, queries: FloatArray) -> None:
    """Refuse a supplied corpus that does not match what the workload declared."""
    if corpus.ndim != 2 or queries.ndim != 2:
        raise ConfigError(
            f"corpus and queries must be 2-D; got {corpus.shape} and {queries.shape}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if corpus.shape[1] != workload.dimension or queries.shape[1] != workload.dimension:
        raise ConfigError(
            f"workload declares dimension {workload.dimension}; corpus has "
            f"{corpus.shape[1]} and queries have {queries.shape[1]}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if workload.k > corpus.shape[0]:
        raise ConfigError(
            f"k={workload.k} exceeds the supplied corpus of {corpus.shape[0]} vectors",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )


class VectorBenchmark:
    """Executes a vector workload against one adapter."""

    def __init__(
        self,
        workload: VectorWorkload,
        corpus: FloatArray | None = None,
        queries: FloatArray | None = None,
    ) -> None:
        """Measure a workload, over a supplied corpus or a seeded synthetic one.

        Passing real vectors is how a verified dataset reaches the measurement.
        Nothing else may supply them: a run that recorded a dataset id while
        measuring generated noise would put a false provenance claim into an
        immutable bundle, which is the one failure this project cannot have.
        """
        self.workload = workload
        if (corpus is None) != (queries is None):
            raise ConfigError(
                "supply both a corpus and a query set, or neither",
                context=ErrorContext(phase=Phase.DATASET_LOAD),
            )
        if corpus is None or queries is None:
            self.corpus, self.queries = generate_corpus(workload)
            self.synthetic = True
        else:
            _check_supplied(workload, corpus, queries)
            self.corpus, self.queries = corpus, queries
            self.synthetic = False
        # The oracle is computed once, by us, from the same bytes the system
        # was given -- never taken from the system's own answer (TRD D6).
        self._ground_truth_ids, self._ground_truth = brute_force_ground_truth(
            self.corpus, self.queries, workload.k, workload.metric
        )

    # ------------------------------------------------------------------ phases

    def load(self, adapter: SystemAdapter) -> float:
        outcome = adapter.load_dataset(self.workload.table_spec(), self.corpus)
        if not outcome.complete:
            raise MeasurementError(
                f"loaded {outcome.rows_loaded} of {outcome.rows_expected} vectors",
                context=ErrorContext(phase=Phase.DATASET_LOAD, system=adapter.system_id),
            )
        return outcome.seconds

    def warm_up(self, adapter: SystemAdapter) -> int:
        """Untimed queries before the measured window.

        Nothing here is recorded. Warm-up that contributed to the reported
        numbers would be a measurement of a half-warm cache.
        """
        count = min(self.workload.warmup_queries, self.queries.shape[0])
        for index in range(count):
            adapter.execute(self._query(index))
        return count

    def measure(self, adapter: SystemAdapter, repetition: int) -> RepetitionResult:
        """One timed pass over the query set."""
        sample = self._sample_size()
        latencies: list[float] = []
        returned: list[list[int]] = []
        errors = 0
        timeouts = 0

        started = time.perf_counter()
        for index in range(sample):
            try:
                result = adapter.execute(self._query(index))
            except SystemUnavailableError:
                # The system is gone. This is not a query that failed, it is
                # the end of the run: continuing would record a dead system as
                # a stream of query errors and let the "did it crash" check
                # pass while it lay dead.
                raise
            except MeasurementError:
                # A timeout is a distinct outcome, not a slow success: it never
                # enters the latency distribution.
                timeouts += 1
                continue
            except BenchError:
                errors += 1
                continue
            latencies.append(result.latency_seconds * 1000.0)
            returned.append(list(result.ids))
        duration = time.perf_counter() - started

        return RepetitionResult(
            repetition=repetition,
            successes=len(latencies),
            errors=errors,
            timeouts=timeouts,
            duration_seconds=duration,
            latency=summarise_latency(latencies),
            recall=self._recall(returned, sample),
        )

    # ------------------------------------------------------------------ recall

    def _recall(self, returned: Sequence[Sequence[int]], sample: int) -> float | None:
        """Recall over the queries that actually returned an answer.

        Returns None rather than 0.0 when nothing came back: a system that
        answered nothing has no measured quality, and zero would read as a
        measurement of terrible quality instead of an absence of one.
        """
        if not returned:
            return None
        k = self.workload.k
        usable = [ids for ids in returned if len(ids) >= k]
        if not usable:
            return 0.0
        ids = np.asarray(usable[: len(usable)], dtype=np.int64)[:, :k]
        corpus_size = self.corpus.shape[0]
        if ids.min() < 0 or ids.max() >= corpus_size:
            # The system returned an id that is not in the corpus. That is a
            # correctness failure, not a quality score.
            raise MeasurementError(
                f"system returned neighbour id outside the corpus of {corpus_size} vectors",
                context=ErrorContext(phase=Phase.MEASUREMENT),
            )
        from theodb_bench.analysis.quality import neighbors_ground_truth

        run_distances = neighbors_ground_truth(
            self.corpus, self.queries[: ids.shape[0]], ids, k, self.workload.metric
        )
        return recall_at_k(self._ground_truth[: ids.shape[0]], run_distances, k)

    # ------------------------------------------------------------------ points

    def run_point(
        self,
        adapter: SystemAdapter,
        index: IndexSpec,
        search: dict[str, Any],
        repetitions: int,
    ) -> PointResult:
        """Build, warm up and measure one configuration."""
        label = build_label(index, search, self.workload.query_cap)
        point = PointResult(label=label, parameters={**index.parameters, **search})

        spec = self.workload.table_spec()
        try:
            # Other configurations' indexes go first: leaving them lets the
            # system choose between two indexes on the same column.
            adapter.drop_indexes(spec)
            build = adapter.build_index(spec, index)
        except UnsupportedCapabilityError as exc:
            point.status = "unsupported"
            point.status_detail = exc.message
            return point

        adapter.set_search_parameters(search)
        self.warm_up(adapter)

        for repetition in range(1, repetitions + 1):
            result = self.measure(adapter, repetition)
            result.build_seconds = build.seconds
            result.index_size_bytes = build.index_size_bytes
            point.repetitions.append(result)
        return point

    def configurations(self) -> list[tuple[IndexSpec, dict[str, Any]]]:
        """Every (index, search parameter) pair this workload declares."""
        return [
            (index, search) for index in self.workload.indexes for search in self.sweep_for(index)
        ]

    def sweep_for(self, index: IndexSpec) -> list[dict[str, Any]]:
        """Search-parameter combinations that apply to an index kind.

        Exact search has nothing to tune, so sweeping it would produce
        duplicate points under different labels.
        """
        return [{}] if index.kind == "none" else self.workload.sweep_points()

    # ------------------------------------------------------------------ helper

    def _sample_size(self) -> int:
        total = int(self.queries.shape[0])
        return min(self.workload.query_cap, total) if self.workload.query_cap else total

    def _query(self, index: int) -> KnnQuery:
        return KnnQuery(
            table=self.workload.table,
            vector=self.queries[index],
            k=self.workload.k,
            metric=self.workload.metric,
        )
