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
    BatchQuery,
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
    k_sweep: tuple[int, ...] = ()
    """Extra values of k to measure. Empty means "just the declared k", so every
    suite written before this existed measures exactly what it measured. The
    graph descent and the rescore pool both scale with k and not the same way: a
    system fast at k=10 can fall over at the k=100 a reranking pipeline asks for."""
    filter_cardinality: int | None = None
    """Partition the corpus into this many tenants and filter every query to one.
    Filtered ANN is the hardest case for a graph index -- the filter can
    disconnect it -- and the easiest to answer fast and wrongly, so the oracle
    filters too."""
    batch_size: int | None = None
    """Probes per round trip. One trip carrying many probes is what an agent's
    step issues, and it is where per-query overhead stops dominating."""
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
        largest = max(self.k_values)
        if largest > self.corpus_size:
            raise ConfigError(
                f"k={largest} exceeds the corpus size {self.corpus_size}; a system that "
                f"returned fewer than k neighbours would be scored as having missed them",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.filter_cardinality is not None:
            if self.filter_cardinality < 1:
                raise ConfigError(
                    f"filter_cardinality must be at least 1, got {self.filter_cardinality}",
                    context=ErrorContext(phase=Phase.PREFLIGHT),
                )
            per_tenant = self.corpus_size // self.filter_cardinality
            if per_tenant < largest:
                raise ConfigError(
                    f"a filter of {self.filter_cardinality} tenants leaves {per_tenant} rows "
                    f"each, fewer than k={largest}. Scoring that as a miss would blame the "
                    f"system for the workload's arithmetic",
                    context=ErrorContext(phase=Phase.PREFLIGHT),
                )
        if self.batch_size is not None:
            if self.batch_size < 1:
                raise ConfigError(
                    f"batch_size must be at least 1, got {self.batch_size}",
                    context=ErrorContext(phase=Phase.PREFLIGHT),
                )
            if self.batch_size > self.query_count:
                raise ConfigError(
                    f"batch_size {self.batch_size} exceeds the {self.query_count} declared "
                    f"queries; a padded batch issues probes the workload never declared",
                    context=ErrorContext(phase=Phase.PREFLIGHT),
                )

    @property
    def k_values(self) -> tuple[int, ...]:
        """Every k this workload measures, the declared one included."""
        return tuple(sorted({self.k, *self.k_sweep}))

    @property
    def operation_count(self) -> int:
        """Operations issued per repetition.

        Batches, not probes: throughput of a batched run is batches per second,
        and counting probes would make a batch of 100 look a hundred times
        faster than it is.
        """
        if self.batch_size is None:
            return self.query_count
        return -(-self.query_count // self.batch_size)

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
    k: int | None = None,
    batch_size: int | None = None,
    filter_cardinality: int | None = None,
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
    if k is not None:
        parts.append(f"[k={k}]")
    if batch_size is not None:
        parts.append(f"[batch={batch_size}]")
    if filter_cardinality is not None:
        parts.append(f"[filter={filter_cardinality}]")
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
        # Computed once at the largest k and sliced per point: recomputing per k
        # would multiply the most expensive step of a run by the length of the
        # sweep, and the top-10 of a top-100 is the top-10 because the ordering
        # is a total order.
        self._largest_k = max(workload.k_values)
        if workload.filter_cardinality is None:
            self._ground_truth_ids, self._ground_truth = self.binding.ground_truth(
                self.queries, self._largest_k, workload.metric
            )
        else:
            self._ground_truth_ids, self._ground_truth = self._filtered_ground_truth()

    def tenant_of(self, row_id: int) -> str | None:
        """Which tenant a corpus row belongs to.

        A deterministic partition by row id rather than a random assignment: two
        runs of the same benchmark must filter the same rows, or they differ in
        the work as well as in the system.
        """
        cardinality = self.workload.filter_cardinality
        if cardinality is None:
            return None
        return f"t{row_id % cardinality}"

    def _tenant_for_query(self, index: int) -> str | None:
        cardinality = self.workload.filter_cardinality
        if cardinality is None:
            return None
        return f"t{index % cardinality}"

    def _filtered_ground_truth(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Exact neighbours *within each query's tenant*.

        Scoring a filtered query against the unfiltered oracle would reward the
        one defect this workload exists to catch: a graph index whose edges cross
        the filter answers fast and wrong, and the neighbours it returns really
        are the corpus's nearest -- they are simply not the tenant's.
        """
        cardinality = self.workload.filter_cardinality
        if cardinality is None:  # unreachable via the caller; keeps the type honest
            raise ConfigError(
                "a filtered oracle needs a filter cardinality",
                context=ErrorContext(phase=Phase.OFFLINE),
            )
        k = self._largest_k
        queries = self.queries
        ids = np.empty((queries.shape[0], k), dtype=np.int64)
        distances = np.empty((queries.shape[0], k), dtype=np.float64)

        rows = np.arange(self.binding.row_count)
        for tenant_index in range(cardinality):
            members = rows[rows % cardinality == tenant_index]
            wanted = [q for q in range(queries.shape[0]) if q % cardinality == tenant_index]
            if not wanted:
                continue
            subset = self.binding.subset(members)
            local_ids, local_distances = subset.ground_truth(
                queries[wanted], k, self.workload.metric
            )
            for position, query_index in enumerate(wanted):
                ids[query_index] = members[local_ids[position]]
                distances[query_index] = local_distances[position]
        return ids, distances

    def _batch(self, batch_index: int, k: int | None = None) -> BatchQuery:
        """One round trip carrying several probes.

        The last batch is short rather than padded: padding would issue probes
        the workload never declared and count them in the throughput.
        """
        size = self.workload.batch_size
        if size is None:
            raise ConfigError(
                "this workload declares no batch size",
                context=ErrorContext(phase=Phase.MEASUREMENT),
            )
        start = batch_index * size
        stop = min(start + size, self._sample_size())
        return BatchQuery(
            table=self.workload.table,
            vectors=tuple(self.queries[i] for i in range(start, stop)),
            k=k if k is not None else self.workload.k,
            metric=self.workload.metric,
            tenants=tuple(self._tenant_for_query(i) for i in range(start, stop)),
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
        k: int | None = None,
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
        measured_k = k if k is not None else self.workload.k
        batch_size = self.workload.batch_size
        operations = sample if batch_size is None else -(-sample // batch_size)

        answers: dict[int, Any] = {}
        latency_ms: dict[int, float] = {}
        outcome: dict[int, str] = {}
        recorder = threading.Lock()

        def issue_batch(client: SystemAdapter, index: int) -> None:
            """One round trip carrying several probes.

            Recorded as one operation with one timing -- throughput of a batched
            run is batches per second -- but with one recall observation per
            probe: averaging them into the batch would let nine wrong answers
            and one right one score like the reverse.
            """
            batch = self._batch(index, measured_k)
            result = client.execute_batch(batch)
            first = index * (batch_size or 1)
            with recorder:
                for offset, ids in enumerate(result.ids):
                    answers[first + offset] = list(ids)
                latency_ms[index] = result.latency_seconds * 1000.0
                outcome[index] = "ok"

        def issue(client: SystemAdapter, index: int) -> None:
            if batch_size is not None:
                try:
                    issue_batch(client, index)
                except (SystemUnavailableError, UnsupportedCapabilityError):
                    # A system with no batch path is *unsupported* at this point,
                    # not slow at it. Answering the batch as N singles would make
                    # every system look like it batches.
                    raise
                except MeasurementError:
                    with recorder:
                        outcome[index] = "timeout"
                except BenchError:
                    with recorder:
                        outcome[index] = "error"
                return
            try:
                result = client.execute(self._query(index, measured_k))
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
            count=operations,
            model=model,
            # `SystemUnavailableError` ends the run rather than becoming a data
            # point: recording a dead system as a stream of query errors would
            # let the "did it crash" check pass while it lay dead.
            fatal=(SystemUnavailableError, UnsupportedCapabilityError),
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
            recall=self._recall(returned, sample, measured_k),
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

    def _recall(
        self, returned: Sequence[Sequence[int]], sample: int, k: int | None = None
    ) -> float | None:
        """Recall over the queries that actually returned an answer.

        Returns None rather than 0.0 when nothing came back: a system that
        answered nothing has no measured quality, and zero would read as a
        measurement of terrible quality instead of an absence of one.
        """
        if not returned:
            return None
        k = k if k is not None else self.workload.k
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
        k: int | None = None,
        label: str | None = None,
    ) -> PointResult:
        """Build, warm up and measure one configuration."""
        label = (
            label
            if label is not None
            else build_label(index, search, self.workload.query_cap, self.workload.load)
        )
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

        try:
            self.warm_up(adapter)
            for repetition in range(1, repetitions + 1):
                result = self.measure(adapter, repetition, make_client, k=k)
                result.build_seconds = build.seconds
                result.index_size_bytes = build.index_size_bytes
                point.repetitions.append(result)
        except UnsupportedCapabilityError as exc:
            # A shape the system has no path for -- a batch probe, say. The point
            # is unsupported rather than slow, and answering a batch as N singles
            # to produce a number would make every system look like it batches.
            point.status = "unsupported"
            point.status_detail = exc.message
            point.repetitions.clear()
        return point

    def points(
        self,
        adapter: SystemAdapter,
        repetitions: int,
        make_client: Callable[[], SystemAdapter] | None = None,
    ) -> list[PointResult]:
        """Every configuration measured, in order."""
        return [
            self.run_point(adapter, index, search, repetitions, make_client, k=k, label=label)
            for label, (index, search), k in self.configurations_with_k()
        ]

    def configurations(self) -> list[tuple[IndexSpec, dict[str, Any]]]:
        """Every (index, search parameter) pair this workload declares."""
        return [
            (index, search) for index in self.workload.indexes for search in self.sweep_for(index)
        ]

    def configurations_with_k(self) -> list[tuple[str, tuple[IndexSpec, dict[str, Any]], int]]:
        """Every configuration, once per declared k, with the label it will carry.

        The k is part of the label because two points measured at different k
        that share one are two numbers a reader cannot tell apart -- the same
        reason the query cap and the load are in there.
        """
        return [
            (
                build_label(
                    index,
                    search,
                    self.workload.query_cap,
                    self.workload.load,
                    k=k if len(self.workload.k_values) > 1 else None,
                    batch_size=self.workload.batch_size,
                    filter_cardinality=self.workload.filter_cardinality,
                ),
                (index, search),
                k,
            )
            for index, search in self.configurations()
            for k in self.workload.k_values
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

    def _query(self, index: int, k: int | None = None) -> KnnQuery:
        return KnnQuery(
            table=self.workload.table,
            vector=self.queries[index],
            k=k if k is not None else self.workload.k,
            metric=self.workload.metric,
            tenant=self._tenant_for_query(index),
        )
