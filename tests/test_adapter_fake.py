"""The fake system exists so the runner's failure handling can be tested.

Its search is exact, so recall is 1.0 unless a fault degrades it on purpose.
Any other value in these tests is a real finding about the code under test.
"""

from __future__ import annotations

import os
import time

import numpy as np
import numpy.typing as npt
import pytest
from theodb_bench.adapters.base import IndexSpec, KnnQuery, VectorTableSpec, execute_batch
from theodb_bench.adapters.fake import FakeAdapter, FakeConfig, Fault
from theodb_bench.errors import (
    AdapterError,
    MeasurementError,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)
from theodb_bench.isolation import find_escapes, online_cpus
from theodb_bench.schemas import validate

SPEC = VectorTableSpec(table="items", dimension=8, metric="l2")


def _corpus(rows: int = 64, dimension: int = 8, seed: int = 7) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, dimension)).astype(np.float32)


def _ready(config: FakeConfig | None = None) -> FakeAdapter:
    adapter = FakeAdapter(config)
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    return adapter


def _loaded(
    config: FakeConfig | None = None, rows: int = 64
) -> tuple[FakeAdapter, npt.NDArray[np.float32]]:
    adapter = _ready(config)
    corpus = _corpus(rows)
    adapter.load_dataset(SPEC, corpus)
    return adapter, corpus


# ------------------------------------------------------------------- contract


def test_capabilities_cover_the_closed_vocabulary() -> None:
    from theodb_bench.adapters.base import CAPABILITIES

    assert set(FakeAdapter().capabilities()) == set(CAPABILITIES)


def test_system_payload_validates_against_the_system_schema() -> None:
    adapter, _ = _loaded()
    validate("system", adapter.system_payload())
    adapter.stop()


def test_an_unsupported_index_is_refused_not_faked() -> None:
    # ivfflat is declared false, so asking for it must produce an explicit
    # unsupported error rather than a silently different measurement.
    adapter, _ = _loaded()
    with pytest.raises(UnsupportedCapabilityError, match="vector_ivfflat"):
        adapter.build_index(SPEC, IndexSpec(kind="ivfflat", parameters={"lists": 10}))
    adapter.stop()


def test_an_unknown_capability_name_is_a_programming_error() -> None:
    with pytest.raises(AdapterError, match="unknown capability"):
        FakeAdapter().supports("teleportation")


def test_context_manager_starts_and_stops() -> None:
    # The assertion here is about the lifecycle, not about capabilities: on
    # entry the system must be ready to serve, and on exit it must be stopped.
    adapter = FakeAdapter()
    with adapter as entered:
        entered.load_dataset(SPEC, _corpus())
        entered.execute(KnnQuery(table="items", vector=_corpus(1)[0], k=1))
    with pytest.raises(SystemUnavailableError, match="not ready"):
        adapter.execute(KnnQuery(table="items", vector=_corpus(1)[0], k=1))


# --------------------------------------------------------------------- search


def test_exact_search_returns_the_true_neighbours() -> None:
    adapter, corpus = _loaded()
    probe = corpus[3]
    result = adapter.execute(KnnQuery(table="items", vector=probe, k=5))
    # A vector is its own nearest neighbour at distance zero.
    assert result.ids[0] == 3
    assert result.distances[0] == pytest.approx(0.0, abs=1e-6)
    assert len(result.ids) == 5
    adapter.stop()


def test_k_larger_than_the_corpus_returns_the_whole_corpus() -> None:
    adapter, _ = _loaded(rows=4)
    result = adapter.execute(KnnQuery(table="items", vector=_corpus(1)[0], k=100))
    assert len(result.ids) == 4
    adapter.stop()


def test_search_is_deterministic_across_repeated_calls() -> None:
    adapter, corpus = _loaded()
    query = KnnQuery(table="items", vector=corpus[10], k=8)
    first = adapter.execute(query)
    second = adapter.execute(query)
    assert first.ids == second.ids
    adapter.stop()


def test_ties_are_broken_by_id_not_by_sort_order() -> None:
    # Without deterministic tie-breaking, equal distances resolve differently
    # between runs and recall becomes irreproducible at the top-k boundary.
    adapter = _ready()
    duplicated = np.tile(np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32), (5, 1))
    adapter.load_dataset(SPEC, duplicated)
    result = adapter.execute(KnnQuery(table="items", vector=duplicated[0], k=3))
    assert result.ids == (0, 1, 2)
    adapter.stop()


def test_querying_an_unloaded_table_is_an_error() -> None:
    adapter = _ready()
    with pytest.raises(AdapterError, match="never loaded"):
        adapter.execute(KnnQuery(table="missing", vector=_corpus(1)[0], k=1))
    adapter.stop()


def test_loading_the_wrong_dimension_is_refused() -> None:
    adapter = _ready()
    with pytest.raises(AdapterError, match="dimension"):
        adapter.load_dataset(SPEC, _corpus(dimension=16))
    adapter.stop()


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_a_vector_is_its_own_nearest_neighbour_under_distance_metrics(metric: str) -> None:
    adapter = _ready()
    spec = VectorTableSpec(table="items", dimension=8, metric=metric)
    corpus = _corpus()
    adapter.load_dataset(spec, corpus)
    result = adapter.execute(KnnQuery(table="items", vector=corpus[0], k=3, metric=metric))
    assert result.ids[0] == 0
    adapter.stop()


def test_inner_product_ranks_by_magnitude_not_by_identity() -> None:
    # Under inner product a vector is NOT its own nearest neighbour: the
    # largest dot product goes to whichever vector projects furthest along the
    # probe. Asserting self-identity here would encode a wrong expectation and
    # then "fix" the adapter to satisfy it.
    adapter = _ready()
    spec = VectorTableSpec(table="items", dimension=8, metric="ip")
    corpus = _corpus()
    adapter.load_dataset(spec, corpus)
    probe = corpus[0]
    result = adapter.execute(KnnQuery(table="items", vector=probe, k=3, metric="ip"))

    expected = np.argsort(-(corpus.astype(np.float64) @ probe.astype(np.float64)), kind="stable")
    assert result.ids[0] == int(expected[0])
    assert list(result.ids) == [int(i) for i in expected[:3]]
    adapter.stop()


def test_an_unknown_metric_is_refused() -> None:
    adapter = _ready()
    spec = VectorTableSpec(table="items", dimension=8, metric="manhattan")
    adapter.load_dataset(spec, _corpus())
    with pytest.raises(AdapterError, match="unknown metric"):
        adapter.execute(KnnQuery(table="items", vector=_corpus(1)[0], k=1))
    adapter.stop()


# --------------------------------------------------------------------- faults


def test_not_ready_fault_prevents_the_run_from_starting() -> None:
    adapter = FakeAdapter(FakeConfig(fault=Fault.NOT_READY))
    adapter.prepare()
    adapter.start()
    with pytest.raises(SystemUnavailableError, match="never became ready"):
        adapter.wait_ready()
    adapter.stop()


def test_crash_fault_stops_serving_after_the_declared_point() -> None:
    adapter, corpus = _loaded(FakeConfig(fault=Fault.CRASH, fail_after_queries=3))
    query = KnnQuery(table="items", vector=corpus[0], k=2)
    for _ in range(3):
        adapter.execute(query)
    with pytest.raises(SystemUnavailableError, match="crashed"):
        adapter.execute(query)
    adapter.stop()


def test_oom_fault_is_distinguishable_from_a_plain_crash() -> None:
    adapter, corpus = _loaded(FakeConfig(fault=Fault.OOM, fail_after_queries=0))
    with pytest.raises(SystemUnavailableError) as excinfo:
        adapter.execute(KnnQuery(table="items", vector=corpus[0], k=2))
    assert excinfo.value.context.details.get("oom") is True
    adapter.stop()


def test_timeout_fault_raises_rather_than_reporting_a_slow_success() -> None:
    adapter, corpus = _loaded(FakeConfig(fault=Fault.TIMEOUT, timeout_seconds=0.01))
    with pytest.raises(MeasurementError, match="budget"):
        adapter.execute(KnnQuery(table="items", vector=corpus[0], k=2))
    adapter.stop()


def test_slow_fault_is_visibly_slower_than_the_baseline() -> None:
    fast, corpus = _loaded()
    query = KnnQuery(table="items", vector=corpus[0], k=4)
    fast.execute(query)
    baseline = fast.execute(query).latency_seconds
    fast.stop()

    slow, _ = _loaded(FakeConfig(fault=Fault.SLOW, slow_multiplier=50.0))
    slow.execute(query)
    degraded = slow.execute(query).latency_seconds
    slow.stop()
    assert degraded > baseline


def test_quality_regression_keeps_latency_and_breaks_answers() -> None:
    # The failure a throughput-only benchmark cannot see.
    honest, corpus = _loaded()
    query = KnnQuery(table="items", vector=corpus[5], k=6)
    truth = honest.execute(query).ids
    honest.stop()

    degraded_adapter, _ = _loaded(FakeConfig(fault=Fault.QUALITY_REGRESSION))
    degraded = degraded_adapter.execute(query).ids
    degraded_adapter.stop()

    assert len(degraded) == len(truth)
    assert degraded != truth


def test_invalid_output_fault_returns_a_structurally_wrong_answer() -> None:
    adapter, corpus = _loaded(FakeConfig(fault=Fault.INVALID_OUTPUT))
    result = adapter.execute(KnnQuery(table="items", vector=corpus[0], k=5))
    assert len(result.ids) != 5
    adapter.stop()


def test_escaped_child_fault_is_detected_by_the_isolation_check() -> None:
    available = sorted(online_cpus())
    if len(available) < 2:
        pytest.skip("needs at least two CPUs for a child to escape to")
    adapter = FakeAdapter(FakeConfig(fault=Fault.ESCAPED_CHILD))
    adapter.prepare()
    adapter.start()
    try:
        adapter.wait_ready()
        deadline = time.monotonic() + 5.0
        escaped: list[int] = []
        while time.monotonic() < deadline:
            escaped = [e.pid for e in find_escapes(os.getpid(), frozenset({available[0]}))]
            if escaped:
                break
            time.sleep(0.05)
        assert escaped, "the adapter's undeclared subprocess was not detected"
    finally:
        adapter.stop()


# ------------------------------------------------------------------- batching


def test_batch_execution_preserves_order() -> None:
    adapter, corpus = _loaded()
    queries = [KnnQuery(table="items", vector=corpus[i], k=1) for i in range(5)]
    results = execute_batch(adapter, queries)
    assert [r.ids[0] for r in results] == list(range(5))
    adapter.stop()


def test_stats_report_what_was_served() -> None:
    adapter, corpus = _loaded()
    adapter.execute(KnnQuery(table="items", vector=corpus[0], k=1))
    stats = adapter.collect_stats()
    assert stats["queries_served"] == 1
    assert stats["tables"] == ["items"]
    adapter.stop()
