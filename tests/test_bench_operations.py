"""Two clocks. Reporting either alone presents deferred work as free."""

from __future__ import annotations

import pytest
from theodb_bench.absent import Absent
from theodb_bench.adapters.fake import FakeAdapter, FakeConfig
from theodb_bench.bench.operations import (
    BACKLOG_DRAIN,
    INSERT_BASELINE,
    INSERT_VECTORIZED,
    UPDATE_SOURCE,
    WORKER_SATURATION,
    OperationsBenchmark,
    OperationsWorkload,
    compare_clocks,
    generate_rows,
)
from theodb_bench.errors import ConfigError


def _workload(**overrides: object) -> OperationsWorkload:
    base: dict[str, object] = {
        "row_count": 20,
        "dimension": 8,
        "freshness_timeout_seconds": 5.0,
        "freshness_poll_seconds": 0.001,
    }
    base.update(overrides)
    return OperationsWorkload(**base)  # type: ignore[arg-type]


def _ready(config: FakeConfig | None = None) -> FakeAdapter:
    adapter = FakeAdapter(config)
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    return adapter


# ------------------------------------------------------------------- workload


def test_unknown_workloads_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown operations workload"):
        _workload(workloads=("teleport",))


def test_an_empty_workload_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least 1"):
        _workload(row_count=0)


def test_rows_are_deterministic_for_a_seed() -> None:
    first = generate_rows(_workload(seed=3))
    second = generate_rows(_workload(seed=3))
    assert [r.text for r in first] == [r.text for r in second]


# --------------------------------------------------------------- both clocks


def test_the_foreground_clock_is_measured_without_waiting_for_freshness() -> None:
    adapter = _ready()
    try:
        result = OperationsBenchmark(_workload()).run(adapter, INSERT_BASELINE)
        assert result.writes == 20
        assert result.write_latency is not None
        assert not isinstance(result.write_latency.p50, Absent)
        # The baseline deliberately does not measure the second clock.
        assert result.freshness_latency is None
    finally:
        adapter.stop()


def test_the_freshness_clock_is_measured_separately_and_is_slower() -> None:
    # The point of the whole surface: the write returns before the derived
    # embedding exists, so the two numbers must differ.
    adapter = _ready(FakeConfig(write_latency_seconds=0.0002, embed_seconds=0.004))
    try:
        result = OperationsBenchmark(_workload()).run(adapter, INSERT_VECTORIZED)
        assert result.write_latency is not None
        assert result.freshness_latency is not None
        write_p50 = result.write_latency.p50
        fresh_p50 = result.freshness_latency.p50
        assert isinstance(write_p50, float) and isinstance(fresh_p50, float)
        assert fresh_p50 > write_p50
    finally:
        adapter.stop()


def test_every_row_becomes_fresh_when_the_worker_keeps_up() -> None:
    adapter = _ready(FakeConfig(embed_seconds=0.0005))
    try:
        result = OperationsBenchmark(_workload()).run(adapter, INSERT_VECTORIZED)
        assert result.stale_rows == 0
    finally:
        adapter.stop()


def test_a_row_that_never_becomes_fresh_is_counted_not_averaged() -> None:
    # Averaging a timeout into the distribution would let a system that lost
    # the row look merely slow.
    adapter = _ready(FakeConfig(embed_seconds=1.0))
    try:
        workload = _workload(row_count=5, freshness_timeout_seconds=0.05)
        result = OperationsBenchmark(workload).run(adapter, INSERT_VECTORIZED)
        assert result.stale_rows > 0
    finally:
        adapter.stop()


# ------------------------------------------------------------------- updates


def test_an_update_invalidates_the_derived_embedding() -> None:
    # A system that did not invalidate would look instantly fresh while
    # serving a stale answer.
    adapter = _ready(FakeConfig(embed_seconds=0.002))
    try:
        result = OperationsBenchmark(_workload(row_count=10)).run(adapter, UPDATE_SOURCE)
        assert result.freshness_latency is not None
        assert result.writes == 10
    finally:
        adapter.stop()


# ------------------------------------------------------------------- backlog


def test_backlog_drain_reports_peak_depth_and_worker_throughput() -> None:
    adapter = _ready(FakeConfig(write_latency_seconds=0.0, embed_seconds=0.001))
    try:
        result = OperationsBenchmark(_workload(row_count=40)).run(adapter, BACKLOG_DRAIN)
        assert result.peak_queue_depth is not None and result.peak_queue_depth > 0
        assert result.worker_throughput is not None
    finally:
        adapter.stop()


def test_an_undrained_backlog_is_reported_rather_than_waited_out() -> None:
    adapter = _ready(FakeConfig(write_latency_seconds=0.0, embed_seconds=1.0))
    try:
        workload = _workload(row_count=10, freshness_timeout_seconds=0.05)
        result = OperationsBenchmark(workload).run(adapter, BACKLOG_DRAIN)
        assert result.stale_rows > 0
        assert result.status_detail is not None
        assert "still queued" in result.status_detail
    finally:
        adapter.stop()


# ---------------------------------------------------------------- saturation


def test_saturation_reports_a_growing_backlog_as_the_finding() -> None:
    # The result is not a percentile: it is that the queue grows without bound.
    adapter = _ready(FakeConfig(write_latency_seconds=0.0, embed_seconds=0.05))
    try:
        workload = _workload(row_count=60, saturation_write_rate=2000.0)
        result = OperationsBenchmark(workload).run(adapter, WORKER_SATURATION)
        assert result.peak_queue_depth is not None and result.peak_queue_depth > 0
        assert result.status_detail is not None
        assert "backlog grew" in result.status_detail
    finally:
        adapter.stop()


# ------------------------------------------------------------ worker accounting


def test_retries_are_counted_rather_than_disappearing() -> None:
    adapter = _ready(FakeConfig(embed_seconds=0.0005, vectorizer_failure_every=3))
    try:
        result = OperationsBenchmark(_workload(row_count=15)).run(adapter, INSERT_VECTORIZED)
        assert result.retries is not None and result.retries > 0
    finally:
        adapter.stop()


def test_an_adapter_without_a_vectorizer_reports_unsupported() -> None:
    adapter = _ready(FakeConfig(capabilities={"vector_exact": True}))
    try:
        result = OperationsBenchmark(_workload()).run(adapter, INSERT_VECTORIZED)
        assert result.status == "unsupported"
        assert result.writes == 0
    finally:
        adapter.stop()


# ------------------------------------------------------------------ comparison


def test_the_comparison_states_what_the_foreground_win_cost() -> None:
    adapter = _ready(FakeConfig(write_latency_seconds=0.0002, embed_seconds=0.002))
    try:
        benchmark = OperationsBenchmark(_workload(row_count=15))
        results = [
            benchmark.run(adapter, INSERT_BASELINE),
            benchmark.run(adapter, INSERT_VECTORIZED),
        ]
        payload = compare_clocks(results)
        assert payload["freshness_p50_ms"] is not None
        assert "presents deferred work as free" in payload["note"]
    finally:
        adapter.stop()


def test_the_comparison_refuses_to_conclude_from_one_clock() -> None:
    adapter = _ready()
    try:
        only_baseline = [OperationsBenchmark(_workload()).run(adapter, INSERT_BASELINE)]
        payload = compare_clocks(only_baseline)
        assert payload["foreground_delta"] is None
        assert "are needed to state" in payload["note"]
    finally:
        adapter.stop()


def test_metric_series_expose_both_clocks() -> None:
    adapter = _ready(FakeConfig(embed_seconds=0.001))
    try:
        result = OperationsBenchmark(_workload(row_count=10)).run(adapter, INSERT_VECTORIZED)
        series = result.metric_series()
        assert any(name.startswith("write_latency_") for name in series)
        assert any(name.startswith("freshness_") for name in series)
    finally:
        adapter.stop()
