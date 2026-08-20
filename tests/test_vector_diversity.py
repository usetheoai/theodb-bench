"""More than one shape of vector query.

Eleven of twelve registered suites asked the same question: top-10 over the whole
corpus, one vector at a time. A system tuned for exactly that shape looks
excellent and tells you nothing about the shapes an application actually issues.

Three are added here, each because it exercises a different part of the engine:

**k.** The graph descent and the rescore pool both scale with k, and they do not
scale the same way. A system that is fast at k=10 can fall over at k=100, which
is the k a reranking pipeline asks for.

**A filter.** `KnnQuery.tenant` existed in the contract with **no
implementation** — the table had no column to filter on. Filtered ANN is the
hardest case for a graph index (the filter can disconnect the graph) and the
easiest to get quietly wrong: a system may satisfy the filter by over-fetching
and dropping, which is correct and slow, or by descending a graph whose edges
cross the filter, which is fast and *wrong*. Only recall against a filtered
oracle tells them apart, so the oracle here filters too.

**Batch.** One round trip carrying many probes is what an agent's step issues,
and it is where per-query overhead stops dominating. Measured apart because a
batch of 10 that takes as long as 10 singles has no batching at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from theodb_bench.adapters.base import KnnQuery, SystemAdapter, VectorTableSpec
from theodb_bench.bench.vector import VectorWorkload

# ------------------------------------------------------------------------- k


def test_a_workload_sweeps_k_and_says_so_in_the_label() -> None:
    """Two points measured at different k that share a label are two numbers a
    reader cannot tell apart."""
    from theodb_bench.adapters.base import IndexSpec
    from theodb_bench.bench.vector import build_label

    labels = {
        build_label(IndexSpec(kind="hnsw", parameters={}), {"ef_search": 64}, None, k=value)
        for value in (1, 10, 100)
    }

    assert len(labels) == 3
    assert any("k=100" in label for label in labels)


def test_k_sweep_produces_one_configuration_per_value() -> None:
    workload = VectorWorkload(
        corpus_size=256, dimension=4, query_count=8, k=10, k_sweep=(1, 10, 100)
    )

    assert sorted(workload.k_values) == [1, 10, 100]


def test_a_k_larger_than_the_corpus_is_refused() -> None:
    """It cannot be answered, and a system that returns fewer than k neighbours
    would be scored as if it had missed them."""
    from theodb_bench.errors import ConfigError

    with pytest.raises(ConfigError, match="exceeds the corpus"):
        VectorWorkload(corpus_size=50, dimension=4, query_count=4, k=10, k_sweep=(10, 100))


def test_the_default_k_sweep_is_just_the_declared_k() -> None:
    """Additive: every existing suite keeps measuring exactly what it measured."""
    assert VectorWorkload(corpus_size=64, dimension=4, query_count=4, k=10).k_values == (10,)


def test_the_oracle_is_computed_at_the_largest_k_and_sliced() -> None:
    """Recomputing ground truth per k would multiply the most expensive step in
    the run by the length of the sweep, and the top-10 of a top-100 is the
    top-10 — as long as the order is the same total order, which it is."""
    workload = VectorWorkload(
        corpus_size=512, dimension=6, query_count=5, k=10, k_sweep=(1, 10, 50)
    )
    benchmark = workload.build(None, None)

    assert benchmark._ground_truth_ids.shape[1] == 50


# -------------------------------------------------------------------- filter


def test_a_filtered_workload_puts_a_filter_column_on_the_table() -> None:
    """The contract had `tenant` and nothing created a column to filter on."""
    spec = VectorTableSpec(table="t", dimension=4, metric="l2", filter_cardinality=8)

    assert spec.filter_cardinality == 8


def test_the_filter_value_is_derived_from_the_query_index_not_random() -> None:
    """Two runs of the same benchmark must issue the same filters, or they differ
    in the work as well as in the system."""
    workload = VectorWorkload(
        corpus_size=256, dimension=4, query_count=16, k=5, filter_cardinality=4
    )
    benchmark = workload.build(None, None)

    first = [benchmark._query(i).tenant for i in range(16)]
    second = [benchmark._query(i).tenant for i in range(16)]

    assert first == second
    assert len(set(first)) == 4


def test_an_unfiltered_workload_still_issues_no_filter() -> None:
    benchmark = VectorWorkload(corpus_size=64, dimension=4, query_count=4, k=2).build(None, None)

    assert benchmark._query(0).tenant is None


def test_recall_for_a_filtered_query_is_scored_against_a_filtered_oracle() -> None:
    """The defect this exists to catch: a graph index whose edges cross the
    filter answers fast and wrong. Scoring it against the *unfiltered* oracle
    would reward exactly that, because the neighbours it returns really are the
    corpus's nearest — they are simply not the tenant's."""
    workload = VectorWorkload(
        corpus_size=400, dimension=8, query_count=6, k=5, filter_cardinality=4
    )
    benchmark = workload.build(None, None)

    ids = benchmark._ground_truth_ids
    for query_index in range(6):
        tenant = benchmark._query(query_index).tenant
        assert tenant is not None
        for neighbour in ids[query_index]:
            assert benchmark.tenant_of(int(neighbour)) == tenant


def test_the_filter_partitions_the_corpus_deterministically() -> None:
    workload = VectorWorkload(
        corpus_size=100, dimension=4, query_count=4, k=2, filter_cardinality=5
    )
    benchmark = workload.build(None, None)

    assignment = [benchmark.tenant_of(i) for i in range(100)]

    assert len(set(assignment)) == 5
    assert assignment == [benchmark.tenant_of(i) for i in range(100)]


def test_a_filter_that_leaves_fewer_rows_than_k_is_refused() -> None:
    """A tenant with three rows cannot answer a top-10, and scoring it as a miss
    would blame the system for the workload's arithmetic."""
    from theodb_bench.errors import ConfigError

    with pytest.raises(ConfigError, match="filter"):
        VectorWorkload(corpus_size=100, dimension=4, query_count=4, k=10, filter_cardinality=50)


# --------------------------------------------------------------------- batch


def test_a_batch_workload_declares_its_batch_size() -> None:
    workload = VectorWorkload(corpus_size=64, dimension=4, query_count=16, k=2, batch_size=8)

    assert workload.batch_size == 8


def test_a_batch_query_carries_many_vectors() -> None:
    workload = VectorWorkload(corpus_size=64, dimension=4, query_count=16, k=2, batch_size=4)
    benchmark = workload.build(None, None)

    batch = benchmark._batch(0)

    assert len(batch.vectors) == 4
    assert batch.k == 2


def test_the_last_batch_is_short_rather_than_padded() -> None:
    """Padding would issue probes the workload never declared and count them in
    the throughput."""
    workload = VectorWorkload(corpus_size=64, dimension=4, query_count=10, k=2, batch_size=4)
    benchmark = workload.build(None, None)

    assert len(benchmark._batch(2).vectors) == 2


def test_the_operation_count_counts_batches_not_probes() -> None:
    """Throughput of a batched run is batches per second; counting probes would
    make a batch of 100 look a hundred times faster than it is."""
    workload = VectorWorkload(corpus_size=64, dimension=4, query_count=100, k=2, batch_size=10)

    assert workload.operation_count == 10


def test_an_unbatched_workload_counts_queries() -> None:
    workload = VectorWorkload(corpus_size=64, dimension=4, query_count=100, k=2)

    assert workload.batch_size is None
    assert workload.operation_count == 100


def test_a_batch_larger_than_the_query_set_is_refused() -> None:
    from theodb_bench.errors import ConfigError

    with pytest.raises(ConfigError, match="batch"):
        VectorWorkload(corpus_size=64, dimension=4, query_count=8, k=2, batch_size=16)


# --------------------------------------------------- the adapter must honour it


def test_the_adapter_builds_a_filter_column_when_the_spec_asks() -> None:
    from theodb_bench.adapters.postgres import PgvectorAdapter

    statements: list[str] = []

    adapter = PgvectorAdapter()
    adapter._execute = lambda sql, parameters=None: statements.append(sql)  # type: ignore[method-assign]
    adapter._fetch_one = lambda sql, parameters=None: (4,)  # type: ignore[method-assign]

    class _Cursor:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def copy(self, sql: str) -> Any:
            statements.append(sql)
            return _Writer()

    class _Writer:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def write_row(self, row: Any) -> None:
            return None

        def write(self, payload: bytes) -> None:
            return None

    adapter._cursor = _Cursor  # type: ignore[method-assign]
    spec = VectorTableSpec(table="t", dimension=4, metric="l2", filter_cardinality=4)

    adapter.load_dataset(spec, np.zeros((4, 4), dtype=np.float32))

    create = next(s for s in statements if "CREATE TABLE" in s)
    assert "tenant" in create


def test_a_filtered_query_reaches_the_sql_as_a_where_clause() -> None:
    from theodb_bench.adapters.postgres import PgvectorAdapter

    adapter = PgvectorAdapter()
    sql = adapter.knn_sql(
        KnnQuery(table="t", vector=np.zeros(4, dtype=np.float32), k=5, tenant="t3")
    )

    assert "WHERE" in sql.upper()
    assert "tenant" in sql


# --------------------------------- the shapes have to become measured points
#
# Declaring k_sweep / batch_size / filter_cardinality and then measuring the same
# single top-k query would be the worst outcome: a bundle that names three shapes
# and contains one.


class _EchoAdapter(SystemAdapter):
    """Returns the first k ids, so recall is computable and k is observable."""

    system_id = "fake"

    def __init__(self) -> None:
        self.ks: list[int] = []
        self.tenants: list[str | None] = []
        self.batches: list[int] = []

    def capabilities(self) -> dict[str, bool]:
        return {"vector_exact": True}

    def prepare(self) -> None: ...
    def start(self) -> None: ...
    def wait_ready(self, timeout_seconds: float = 60.0) -> None: ...
    def stop(self) -> None: ...
    def cleanup(self) -> None: ...

    def load_dataset(self, spec: Any, vectors: Any) -> Any:
        from theodb_bench.adapters.base import LoadOutcome

        rows = int(np.asarray(vectors).shape[0])
        return LoadOutcome(seconds=0.0, rows_loaded=rows, rows_expected=rows)

    def build_index(self, spec: Any, index: Any) -> Any:
        from theodb_bench.adapters.base import BuildOutcome

        return BuildOutcome(seconds=0.0, index_size_bytes=None, parameters_in_force={})

    def execute(self, query: KnnQuery) -> Any:
        from theodb_bench.adapters.base import KnnResult

        self.ks.append(query.k)
        self.tenants.append(query.tenant)
        return KnnResult(
            ids=tuple(range(query.k)),
            distances=tuple(0.0 for _ in range(query.k)),
            latency_seconds=0.0001,
        )

    def execute_batch(self, query: Any) -> Any:
        from theodb_bench.adapters.base import BatchResult

        self.batches.append(len(query.vectors))
        return BatchResult(
            ids=tuple(tuple(range(query.k)) for _ in query.vectors),
            distances=tuple(tuple(0.0 for _ in range(query.k)) for _ in query.vectors),
            latency_seconds=0.001,
        )

    def collect_stats(self) -> dict[str, Any]:
        return {}

    def export_config(self) -> dict[str, Any]:
        return {}


def test_a_k_sweep_produces_one_point_per_k() -> None:
    workload = VectorWorkload(
        corpus_size=256, dimension=4, query_count=8, k=10, k_sweep=(1, 10, 50)
    )
    benchmark = workload.build(None, None)

    labels = [label for label, _, _ in benchmark.configurations_with_k()]

    assert len(set(labels)) == 3


def test_each_k_point_actually_issues_that_k() -> None:
    workload = VectorWorkload(
        corpus_size=256, dimension=4, query_count=8, k=10, k_sweep=(1, 10, 50)
    )
    benchmark = workload.build(None, None)
    adapter = _EchoAdapter()

    benchmark.points(adapter, repetitions=1)

    assert set(adapter.ks) == {1, 10, 50}


def test_a_filtered_workload_issues_a_tenant_on_every_query() -> None:
    workload = VectorWorkload(
        corpus_size=400, dimension=4, query_count=8, k=5, filter_cardinality=4
    )
    benchmark = workload.build(None, None)
    adapter = _EchoAdapter()

    benchmark.points(adapter, repetitions=1)

    assert all(t is not None for t in adapter.tenants)
    assert len(set(adapter.tenants)) == 4


def test_a_batched_workload_issues_batches_not_singles() -> None:
    workload = VectorWorkload(corpus_size=256, dimension=4, query_count=20, k=5, batch_size=5)
    benchmark = workload.build(None, None)
    adapter = _EchoAdapter()

    result = benchmark.points(adapter, repetitions=1)

    assert adapter.batches == [5, 5, 5, 5]
    assert adapter.ks == []
    assert result[0].repetitions[0].successes == 4


def test_a_batched_workload_still_scores_recall_per_probe() -> None:
    """A batch that answered ten probes has ten recall observations, not one.
    Averaging them into the batch would let a batch with nine wrong answers and
    one right one score the same as the reverse."""
    workload = VectorWorkload(corpus_size=256, dimension=4, query_count=20, k=5, batch_size=5)
    benchmark = workload.build(None, None)

    result = benchmark.points(_EchoAdapter(), repetitions=1)

    assert result[0].repetitions[0].recall is not None


def test_an_adapter_without_batch_support_is_refused_not_serialised() -> None:
    """Answering a batch as N singles and reporting it as a batch would make
    every system look like it batches."""
    from theodb_bench.errors import UnsupportedCapabilityError

    class _NoBatch(_EchoAdapter):
        def execute_batch(self, query: Any) -> Any:
            raise UnsupportedCapabilityError(
                "no batch path",
                context=__import__("theodb_bench.errors", fromlist=["ErrorContext"]).ErrorContext(
                    phase=__import__("theodb_bench.errors", fromlist=["Phase"]).Phase.MEASUREMENT
                ),
            )

    benchmark = VectorWorkload(
        corpus_size=256, dimension=4, query_count=20, k=5, batch_size=5
    ).build(None, None)

    result = benchmark.points(_NoBatch(), repetitions=1)

    assert result[0].status == "unsupported"


# ------------------------------------------------- the batch SQL, on the adapter


def _batch_adapter() -> tuple[Any, list[tuple[str, Any]]]:
    from theodb_bench.adapters.postgres import TheoDBAdapter

    seen: list[tuple[str, Any]] = []
    adapter = TheoDBAdapter()

    def fetch_all(sql: str, parameters: Any = None) -> list[tuple[Any, ...]]:
        seen.append((sql, parameters))
        return [(0, 1, 0.5), (0, 2, 0.6), (1, 3, 0.1), (1, 4, 0.2)]

    adapter._fetch_all = fetch_all  # type: ignore[method-assign]
    return adapter, seen


def test_a_batch_is_one_round_trip() -> None:
    """The entire point. Looping over `execute` here would report round-trip
    savings that never happened."""
    from theodb_bench.adapters.base import BatchQuery

    adapter, seen = _batch_adapter()

    adapter.execute_batch(
        BatchQuery(table="t", vectors=tuple(np.zeros(4, dtype=np.float32) for _ in range(8)), k=2)
    )

    assert len(seen) == 1


def test_each_probe_keeps_its_own_limit() -> None:
    """A single ORDER BY over the union would return the k best *across* probes
    rather than k for each — the answer would be wrong, not merely differently
    shaped."""
    from theodb_bench.adapters.base import BatchQuery

    adapter, seen = _batch_adapter()

    adapter.execute_batch(BatchQuery(table="t", vectors=(np.zeros(4, dtype=np.float32),) * 2, k=2))

    assert seen[0][0].count("LIMIT 2") == 2


def test_a_batch_result_is_grouped_back_per_probe() -> None:
    from theodb_bench.adapters.base import BatchQuery

    adapter, _ = _batch_adapter()

    result = adapter.execute_batch(
        BatchQuery(table="t", vectors=(np.zeros(4, dtype=np.float32),) * 2, k=2)
    )

    assert result.ids == ((1, 2), (3, 4))


def test_a_filtered_batch_carries_each_probe_s_own_tenant() -> None:
    from theodb_bench.adapters.base import BatchQuery

    adapter, seen = _batch_adapter()

    adapter.execute_batch(
        BatchQuery(
            table="t",
            vectors=(np.zeros(4, dtype=np.float32),) * 2,
            k=2,
            tenants=("t1", "t2"),
        )
    )

    assert "'t1'" in seen[0][0]
    assert "'t2'" in seen[0][0]


def test_an_empty_batch_does_not_reach_the_server() -> None:
    from theodb_bench.adapters.base import BatchQuery

    adapter, seen = _batch_adapter()

    result = adapter.execute_batch(BatchQuery(table="t", vectors=(), k=2))

    assert seen == []
    assert result.ids == ()


def test_the_base_contract_refuses_a_batch_rather_than_looping() -> None:
    """Every adapter inherits a refusal, so a system that gains a batch path
    gains it deliberately."""
    import inspect

    from theodb_bench.adapters.base import SystemAdapter

    source = inspect.getsource(SystemAdapter.execute_batch)

    assert "UnsupportedCapabilityError" in source
    assert "execute" not in source.split('"""')[2]  # the body loops over nothing
