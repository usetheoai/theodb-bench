"""A vector workload measured by a client population rather than one client.

The measure loop issued queries one at a time from one connection. That number
answers "how fast is it when idle", and every published throughput figure for a
database is about the other case.

Wiring the load engine in has one property worth pinning above all others: the
existing single-client behaviour must be exactly what a `LoadModel()` default
produces. A change that silently switched every registered suite to a different
regime would make today's numbers incomparable with yesterday's, and the harness
would have broken its own baseline rule to gain a feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from theodb_bench.adapters.base import (
    BuildOutcome,
    IndexSpec,
    KnnQuery,
    KnnResult,
    LoadOutcome,
    SystemAdapter,
    VectorTableSpec,
)
from theodb_bench.bench.vector import VectorWorkload
from theodb_bench.load import LoadModel


class _CountingAdapter(SystemAdapter):
    """Answers instantly and records which connection served each query."""

    system_id = "fake"
    built = 0

    def __init__(self) -> None:
        type(self).built += 1
        self.connection_id = type(self).built
        self.served = 0

    def capabilities(self) -> dict[str, bool]:
        return {"vector_exact": True}

    def prepare(self) -> None:
        return None

    def start(self) -> None:
        return None

    def wait_ready(self, timeout_seconds: float = 60.0) -> None:
        return None

    def load_dataset(self, spec: VectorTableSpec, vectors: Any) -> LoadOutcome:
        rows = int(np.asarray(vectors).shape[0])
        return LoadOutcome(seconds=0.01, rows_loaded=rows, rows_expected=rows)

    def build_index(self, spec: VectorTableSpec, index: IndexSpec) -> BuildOutcome:
        return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})

    def execute(self, query: KnnQuery) -> KnnResult:
        self.served += 1
        return KnnResult(ids=(0, 1), distances=(0.0, 0.1), latency_seconds=0.0005)

    def collect_stats(self) -> dict[str, Any]:
        return {}

    def export_config(self) -> dict[str, Any]:
        return {}

    def stop(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


def _workload(**kwargs: Any) -> VectorWorkload:
    return VectorWorkload(corpus_size=64, dimension=4, query_count=32, k=2, **kwargs)


def test_the_default_is_still_one_client_closed_loop() -> None:
    """The baseline rule: adding the capability must not move the default."""
    assert _workload().load.clients == 1
    assert _workload().load.is_closed_loop


def test_a_workload_can_declare_a_client_population() -> None:
    workload = _workload(load=LoadModel(clients=8))

    assert workload.load.clients == 8


def test_concurrent_measurement_uses_one_connection_per_client() -> None:
    """Sharing a psycopg connection across threads measures the lock."""
    _CountingAdapter.built = 0
    workload = _workload(load=LoadModel(clients=4))
    benchmark = workload.build(None, None)
    adapter = _CountingAdapter()

    benchmark.measure(adapter, repetition=1, make_client=_CountingAdapter)

    # One per client, plus the adapter the caller already had.
    assert _CountingAdapter.built == 5


def test_every_query_is_still_issued_exactly_once() -> None:
    _CountingAdapter.built = 0
    workload = _workload(load=LoadModel(clients=4))
    benchmark = workload.build(None, None)

    result = benchmark.measure(_CountingAdapter(), repetition=1, make_client=_CountingAdapter)

    assert result.successes == 32


def test_a_concurrent_repetition_still_scores_recall() -> None:
    """Quality is not traded for load. A throughput number without a recall
    number beside it is the oldest way to look fast."""
    _CountingAdapter.built = 0
    workload = _workload(load=LoadModel(clients=4))
    benchmark = workload.build(None, None)

    result = benchmark.measure(_CountingAdapter(), repetition=1, make_client=_CountingAdapter)

    assert result.recall is not None


def test_the_repetition_carries_the_load_summary() -> None:
    """Throughput under contention means nothing without the shape of the load
    that produced it — how many clients, and at what rate."""
    _CountingAdapter.built = 0
    workload = _workload(load=LoadModel(clients=4, arrival_rate=2000.0))
    benchmark = workload.build(None, None)

    result = benchmark.measure(_CountingAdapter(), repetition=1, make_client=_CountingAdapter)

    assert result.load is not None
    assert result.load["clients"] == 4
    assert result.load["arrival_rate_per_second"] == 2000.0
    assert result.load["loop"] == "open"
    assert result.load["response_p99_ms"] is not None


def test_a_single_client_run_records_no_load_summary_it_cannot_support() -> None:
    """A closed loop with one client has no queueing to report, and reporting a
    zero would read as a measured absence of queueing rather than as a regime
    where the question does not arise."""
    _CountingAdapter.built = 0
    benchmark = _workload().build(None, None)

    result = benchmark.measure(_CountingAdapter(), repetition=1)

    assert result.load is None or result.load["arrival_rate_per_second"] is None


def test_the_point_label_names_the_load() -> None:
    """A sweep over client counts whose points all share a label is a sweep that
    flattens onto itself — the same defect the query-cap label exists to avoid."""
    from theodb_bench.bench.vector import build_label

    label = build_label(
        IndexSpec(kind="hnsw", parameters={}),
        {"ef_search": 64},
        None,
        load=LoadModel(clients=16, arrival_rate=500.0),
    )

    assert "16" in label
    assert "500" in label


def test_a_closed_loop_label_is_unchanged() -> None:
    """Existing labels must not move, or every stored baseline stops matching."""
    from theodb_bench.bench.vector import build_label

    assert build_label(
        IndexSpec(kind="hnsw", parameters={}), {"ef_search": 64}, None, load=LoadModel()
    ) == build_label(IndexSpec(kind="hnsw", parameters={}), {"ef_search": 64}, None)


# ------------------------------------- an ordering that is load-bearing
#
# `SystemUnavailableError` is a subclass of `BenchError`. Catching the general
# case first records a dead system as a stream of query errors and lets the "did
# it crash" check pass while it lay dead. Routing the measure loop through the
# load engine reintroduced exactly that, and the runner's crash test caught it —
# the same shape as psycopg's `QueryCanceled` being an `OperationalError`.


def test_a_dead_system_ends_the_run_instead_of_becoming_error_counts() -> None:
    from theodb_bench.errors import ErrorContext, Phase, SystemUnavailableError

    class _Dying(_CountingAdapter):
        def execute(self, query: KnnQuery) -> KnnResult:
            raise SystemUnavailableError(
                "connection is gone",
                context=ErrorContext(phase=Phase.MEASUREMENT, system="fake"),
            )

    benchmark = _workload().build(None, None)

    with pytest.raises(SystemUnavailableError):
        benchmark.measure(_Dying(), repetition=1)


def test_the_same_holds_under_concurrency() -> None:
    """The concurrent path re-raises across threads, or a dead system under load
    would be the one case that slips through."""
    from theodb_bench.errors import ErrorContext, Phase, SystemUnavailableError

    class _Dying(_CountingAdapter):
        def execute(self, query: KnnQuery) -> KnnResult:
            raise SystemUnavailableError(
                "connection is gone",
                context=ErrorContext(phase=Phase.MEASUREMENT, system="fake"),
            )

    benchmark = _workload(load=LoadModel(clients=4)).build(None, None)

    with pytest.raises(SystemUnavailableError):
        benchmark.measure(_Dying(), repetition=1, make_client=_Dying)


def test_an_ordinary_query_failure_is_still_counted_not_fatal() -> None:
    """Only a gone system ends the run. A query that fails is a measurement of a
    system answering badly, which is a finding rather than a stop."""
    from theodb_bench.errors import ErrorContext, MeasurementError, Phase

    class _Timeouts(_CountingAdapter):
        def execute(self, query: KnnQuery) -> KnnResult:
            raise MeasurementError(
                "statement timeout",
                context=ErrorContext(phase=Phase.MEASUREMENT, system="fake"),
            )

    result = _workload().build(None, None).measure(_Timeouts(), repetition=1)

    assert result.timeouts == 32
    assert result.successes == 0


def test_concurrency_without_a_connection_factory_is_refused() -> None:
    """Serialising eight declared clients on one connection would measure the
    lock and report it as the database."""
    from theodb_bench.errors import ConfigError

    benchmark = _workload(load=LoadModel(clients=8)).build(None, None)

    with pytest.raises(ConfigError, match="connection per client"):
        benchmark.measure(_CountingAdapter(), repetition=1)


# ---------------------------------------- the bundle has to say which regime ran
#
# `benchmark_payload` declared `"loop": "closed"` as a literal. That was true of
# every workload when it was written and becomes a false provenance claim the
# moment one declares an arrival rate — the bundle would name a regime the run
# did not use, which is the one class of error this project cannot have.


def test_the_payload_declares_the_regime_that_actually_ran() -> None:
    closed = _workload().benchmark_payload()
    opened = _workload(load=LoadModel(clients=8, arrival_rate=400.0)).benchmark_payload()

    assert closed["workload"]["loop"] == "closed"
    assert opened["workload"]["loop"] == "open"


def test_the_payload_carries_the_client_count() -> None:
    payload = _workload(load=LoadModel(clients=16)).benchmark_payload()

    assert payload["workload"]["clients"] == 16


def test_the_payload_states_the_arrival_rate_or_its_absence() -> None:
    """Absent, not zero — a closed loop has no rate, and zero is a rate."""
    assert _workload().benchmark_payload()["workload"]["arrival_rate_per_second"] is None
    assert (
        _workload(load=LoadModel(clients=4, arrival_rate=250.0)).benchmark_payload()["workload"][
            "arrival_rate_per_second"
        ]
        == 250.0
    )
