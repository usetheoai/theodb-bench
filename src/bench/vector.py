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
import threading
from collections.abc import Callable, Sequence
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
from theodb_bench.analysis.quality import recall_at_k
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.bench.corpus import CorpusBinding, ResidentCorpus, binding_for
from theodb_bench.errors import (
    BenchError,
    ConfigError,
    ErrorContext,
    MeasurementError,
    Phase,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)
from theodb_bench.load import LoadModel, run_load, summarise_load
from theodb_bench.streaming import CorpusSource

DEFAULT_TABLE: Final[str] = "bench_vectors"

FloatArray = npt.NDArray[np.float32]

#: What a vector benchmark accepts as its corpus: an array when it fits, a source
#: read in ranges when it does not (`bench/corpus.py`).
VectorCorpus = FloatArray | CorpusSource


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
    load: LoadModel = field(default_factory=LoadModel)
    """How the queries are issued. The default is one client, closed loop --
    exactly what this workload did before the load engine existed, so adding the
    capability does not move any existing number."""
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

    # ----------------------------------------------------- Workload protocol

    def build(self, corpus: VectorCorpus | None, queries: FloatArray | None) -> VectorBenchmark:
        if corpus is None or queries is None:
            corpus, queries = generate_corpus(self)
        return VectorBenchmark(self, corpus, queries)

    def benchmark_payload(self) -> dict[str, Any]:
        return {
            "workload": {
                "type": "ann",
                # Read from the load model rather than written as a literal: it
                # was true of every workload when it was written, and becomes a
                # false provenance claim the moment one declares an arrival rate.
                "loop": "closed" if self.load.is_closed_loop else "open",
                "clients": self.load.clients,
                "arrival_rate_per_second": self.load.arrival_rate,
                "k": [self.k],
                "operation_count": self.query_count,
            },
            "quality": {"metric": "recall", "ground_truth": "computed"},
            "parameters": {name: list(values) for name, values in self.search_sweep.items()},
        }

    def expected_operations(self, measured_points: int, repetitions: int) -> int:
        sample = self.query_cap or self.query_count
        return measured_points * repetitions * sample

    @property
    def warmup_operations(self) -> int:
        return self.warmup_queries

    def quality_was_reported(self, points: list[Any]) -> bool:
        return any(
            repetition.recall is not None for point in points for repetition in point.repetitions
        )


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
    load: dict[str, Any] | None = None
    """The shape of the load that produced these numbers -- clients, arrival
    rate, and the two latencies kept apart. `None` for a single-client closed
    loop, where there is no queueing to report and a zero would read as a
    measured absence rather than a regime where the question does not arise."""

    #: Latency in milliseconds per query id, for the queries that answered.
    #:
    #: Kept, rather than summarised away, because a paired test between two
    #: systems needs the same query on both sides (I14) and the summary cannot
    #: supply that. Keyed by query id and not by position: the loop skips queries
    #: that errored or timed out, so position *i* is not query *i*, and pairing by
    #: position would misalign every sample after the first timeout.
    latency_by_query: dict[int, float] = field(default_factory=dict)

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


def build_label(
    index: IndexSpec,
    search: dict[str, Any],
    query_cap: int | None,
    load: LoadModel | None = None,
) -> str:
    """A label a reader can map back to a configuration.

    A reduced query sample is part of the label. Hiding it would let two points
    measured over different sample sizes sit in the same table as equals.

    So is the load, for the same reason: a sweep over client counts whose points
    all carry the same label is a sweep that flattens onto itself. A default
    closed loop adds nothing, so every label written before this existed still
    reads the same and every stored baseline still matches.
    """
    parts = [index.label()]
    if search:
        parts.append(" ".join(f"{key}={value}" for key, value in sorted(search.items())))
    if query_cap is not None:
        parts.append(f"[q={query_cap}]")
    if load is not None and not (load.clients == 1 and load.is_closed_loop):
        rate = "" if load.arrival_rate is None else f" @{load.arrival_rate:g}/s"
        parts.append(f"[c={load.clients}{rate}]")
    return " ".join(parts)


def _check_supplied(workload: VectorWorkload, binding: CorpusBinding, queries: FloatArray) -> None:
    """Refuse a supplied corpus that does not match what the workload declared.

    Checked through the binding rather than an array shape: a streamed corpus has
    a row count and a dimension without ever being materialised, and these are
    the only two properties this validation ever needed.
    """
    if queries.ndim != 2:
        raise ConfigError(
            f"queries must be 2-D; got {queries.shape}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if binding.dimension != workload.dimension or queries.shape[1] != workload.dimension:
        raise ConfigError(
            f"workload declares dimension {workload.dimension}; corpus has "
            f"{binding.dimension} and queries have {queries.shape[1]}",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )
    if workload.k > binding.row_count:
        raise ConfigError(
            f"k={workload.k} exceeds the supplied corpus of {binding.row_count} vectors",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )


class VectorBenchmark:
    """Executes a vector workload against one adapter."""

    def __init__(
        self,
        workload: VectorWorkload,
        corpus: VectorCorpus | None = None,
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
            vectors, self.queries = generate_corpus(workload)
            self.binding: CorpusBinding = ResidentCorpus(vectors)
            self.synthetic = True
        else:
            self.binding = binding_for(corpus)
            self.queries = queries
            _check_supplied(workload, self.binding, queries)
            self.synthetic = False
        # The oracle is computed once, by us, from the same bytes the system
        # was given -- never taken from the system's own answer (TRD D6). Which
        # oracle runs follows the corpus shape, and the two are pinned equivalent
        # in `tests/test_corpus_binding.py`.
        self._ground_truth_ids, self._ground_truth = self.binding.ground_truth(
            self.queries, workload.k, workload.metric
        )

    @property
    def corpus(self) -> FloatArray:
        """The corpus as an array, for callers that can hold it.

        A streamed corpus refuses rather than assembling itself: materialising
        20 000 000 x 128 float32 to satisfy an attribute access is 10.2 GB, and
        it is exactly the allocation the streaming path exists to avoid.
        """
        if isinstance(self.binding, ResidentCorpus):
            return self.binding.vectors
        raise ConfigError(
            f"this benchmark streams its corpus of {self.binding.row_count} vectors and "
            f"cannot hand it over as one array; read it in ranges through `binding`",
            context=ErrorContext(phase=Phase.DATASET_LOAD),
        )

    # ------------------------------------------------------------------ phases

    def load(self, adapter: SystemAdapter) -> float:
        outcome = self.binding.load(adapter, self.workload.table_spec())
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

    def measure(
        self,
        adapter: SystemAdapter,
        repetition: int,
        make_client: Callable[[], SystemAdapter] | None = None,
    ) -> RepetitionResult:
        """One timed pass over the query set, under the declared load.

        Routed through one engine whatever the regime, so the single-client path
        is not a second implementation that can drift from the concurrent one.
        A default `LoadModel()` is one client issuing the next query when the
        previous returns -- which is exactly what this method did before the
        engine existed, and is why every existing number stays comparable.

        `make_client` supplies a connection per client. Concurrency without it
        is refused rather than silently serialised on a shared connection: that
        would measure the lock and report it as the database.
        """
        sample = self._sample_size()
        model = self.workload.load

        answers: dict[int, Any] = {}
        latency_ms: dict[int, float] = {}
        outcome: dict[int, str] = {}
        recorder = threading.Lock()

        def issue(client: SystemAdapter, index: int) -> None:
            try:
                result = client.execute(self._query(index))
            except SystemUnavailableError:
                # Re-raised before the BenchError clause below, because it is a
                # subclass of it. Catching it there would record a dead system as
                # a stream of query errors and let the "did it crash" check pass
                # while it lay dead -- the ordering is load-bearing, exactly like
                # psycopg's QueryCanceled being an OperationalError.
                raise
            except MeasurementError:
                # A timeout is a distinct outcome, not a slow success: it never
                # enters the latency distribution.
                with recorder:
                    outcome[index] = "timeout"
                return
            except BenchError:
                with recorder:
                    outcome[index] = "error"
                return
            with recorder:
                answers[index] = list(result.ids)
                # The adapter's own timing, not the wall clock around it: under
                # concurrency the wall clock includes the client's own queueing,
                # which the load summary reports separately and by name.
                latency_ms[index] = result.latency_seconds * 1000.0
                outcome[index] = "ok"

        clients, closer = self._client_pool(adapter, make_client, model)
        load_result = run_load(
            make_client=clients,
            issue=issue,
            count=sample,
            model=model,
            # `SystemUnavailableError` ends the run rather than becoming a data
            # point: recording a dead system as a stream of query errors would
            # let the "did it crash" check pass while it lay dead.
            fatal=(SystemUnavailableError,),
            close_client=closer,
        )

        returned = [answers[i] for i in sorted(answers)]
        latencies = [latency_ms[i] for i in sorted(latency_ms)]

        return RepetitionResult(
            repetition=repetition,
            successes=len(latencies),
            errors=sum(1 for v in outcome.values() if v == "error"),
            timeouts=sum(1 for v in outcome.values() if v == "timeout"),
            duration_seconds=load_result.duration_seconds,
            latency=summarise_latency(latencies),
            recall=self._recall(returned, sample),
            latency_by_query=dict(latency_ms),
            load=self._load_summary(load_result, model),
        )

    def _client_pool(
        self,
        adapter: SystemAdapter,
        make_client: Callable[[], SystemAdapter] | None,
        model: LoadModel,
    ) -> tuple[Callable[[], SystemAdapter], Callable[[SystemAdapter], None] | None]:
        """The per-client connections, and how to dispose of them.

        One client reuses the adapter the caller already opened -- opening a
        second connection to do the same work it did before would change the
        measurement to gain nothing.
        """
        if model.clients == 1:
            return (lambda: adapter), None
        if make_client is None:
            raise ConfigError(
                f"this workload declares {model.clients} clients and no way to open a "
                f"connection per client. Serialising them on one connection would measure "
                f"the lock and report it as the database.",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )

        def build() -> SystemAdapter:
            client = make_client()
            client.prepare()
            client.start()
            client.wait_ready()
            return client

        return build, lambda client: client.stop()

    @staticmethod
    def _load_summary(load_result: Any, model: LoadModel) -> dict[str, Any] | None:
        """The load's shape, when there is one worth reporting.

        A single-client closed loop has no queueing, and a zero there would read
        as a measured absence of queueing rather than as a regime in which the
        question does not arise -- the same distinction the four absence kinds
        exist to keep.
        """
        if model.clients == 1 and model.is_closed_loop:
            return None
        return summarise_load(load_result)

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
        # From the binding, not from an array: a streamed corpus knows its row
        # count without ever being materialised, and reaching for the array here
        # refused a 20 000 000-vector run *after* it had loaded and queried.
        corpus_size = self.binding.row_count
        if ids.min() < 0 or ids.max() >= corpus_size:
            # The system returned an id that is not in the corpus. That is a
            # correctness failure, not a quality score.
            raise MeasurementError(
                f"system returned neighbour id outside the corpus of {corpus_size} vectors",
                context=ErrorContext(phase=Phase.MEASUREMENT),
            )
        run_distances = self.binding.returned_distances(
            self.queries[: ids.shape[0]], ids, k, self.workload.metric
        )
        return float(recall_at_k(self._ground_truth[: ids.shape[0]], run_distances, k))

    # ------------------------------------------------------------------ points

    def run_point(
        self,
        adapter: SystemAdapter,
        index: IndexSpec,
        search: dict[str, Any],
        repetitions: int,
        make_client: Callable[[], SystemAdapter] | None = None,
    ) -> PointResult:
        """Build, warm up and measure one configuration."""
        label = build_label(index, search, self.workload.query_cap, self.workload.load)
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
        # B-060 — record what the server has IN FORCE next to what was requested.
        #
        # The two are not always the same, and the difference is not hypothetical: `probes` is
        # clamped to the list count derived from the real row count, so a request of 10000 on a
        # 10k-row table is sent as 50. `point.parameters` above is built from the REQUEST, so
        # without this the bundle would report an operating point that never existed.
        #
        # Keyed by GUC name (`ivfflat.probes`), which keeps it unambiguous against the logical
        # request key (`probes`) and needs no schema change: `points[].parameters` is already
        # declared as an open object of scalars, so this uses the field as intended rather than
        # adding a property to a versioned schema.
        point.parameters.update(adapter.effective_search_parameters())
        self.warm_up(adapter)

        for repetition in range(1, repetitions + 1):
            result = self.measure(adapter, repetition, make_client)
            result.build_seconds = build.seconds
            result.index_size_bytes = build.index_size_bytes
            point.repetitions.append(result)
        return point

    def points(
        self,
        adapter: SystemAdapter,
        repetitions: int,
        make_client: Callable[[], SystemAdapter] | None = None,
    ) -> list[PointResult]:
        """Every configuration measured, in order."""
        return [
            self.run_point(adapter, index, search, repetitions, make_client)
            for index, search in self.configurations()
        ]

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
