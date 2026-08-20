"""Issuing work the way a real client population does.

The harness measured one client at a time: `for i in range(sample):
adapter.execute(...)`. That is a *closed loop with one client*, and it answers a
question nobody asks of a database — how fast is it when nothing else is
happening. Throughput under contention is where a database actually lives, and no
number here could describe it.

Two things have to exist for that, and the second is the one that makes the
measurement honest.

**Concurrency.** N clients, each with its own connection, because a psycopg
connection is not safe to share and sharing one would measure the lock rather
than the database. Threads rather than asyncio: the driver releases the GIL on
network I/O, so threads are what the installed dependency already supports.

**An arrival rate, and latency measured from it.** In a closed loop a stalled
system receives *fewer* requests, so the stall never enters the latency
distribution — the system looks fine and the queue is invisible. This is
coordinated omission (Tene). The fix is to schedule each request in advance and
measure from the moment it *should* have been issued, not the moment it was.

So every request carries both numbers and they are never conflated:

    response time = finished - scheduled   (includes queueing; the honest one)
    service  time = finished - started     (what the server took)

Reporting only service time is precisely the omission. Reporting both makes the
queue visible, and their difference *is* the queueing delay.
"""

from __future__ import annotations

import threading
import time
from itertools import pairwise
from typing import Any

import pytest
from theodb_bench.load import (
    LoadModel,
    arrival_schedule,
    run_load,
    summarise_load,
)

# ------------------------------------------------------------------ the model


def test_a_closed_loop_is_the_default_and_declares_it() -> None:
    """One client, no arrival rate: exactly what the harness did before. It stays
    reachable so the change is additive rather than a silent switch of regime."""
    model = LoadModel()

    assert model.clients == 1
    assert model.arrival_rate is None
    assert model.is_closed_loop


def test_an_arrival_rate_makes_it_open_loop() -> None:
    model = LoadModel(clients=8, arrival_rate=500.0)

    assert not model.is_closed_loop


def test_zero_clients_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LoadModel(clients=0)


def test_a_non_positive_arrival_rate_is_refused() -> None:
    """A rate of zero would schedule the first request at infinity, and a
    negative one is not a rate. Both are refused rather than clamped."""
    with pytest.raises(ValueError, match="positive"):
        LoadModel(clients=2, arrival_rate=0.0)


# --------------------------------------------------------------- the schedule


def test_the_schedule_is_seeded_and_reproducible() -> None:
    """Two runs of the same benchmark must differ only in the system under test.
    An unseeded arrival process would make them differ in the load too."""
    first = arrival_schedule(count=50, rate=100.0, seed=7)
    second = arrival_schedule(count=50, rate=100.0, seed=7)

    assert first == second


def test_a_different_seed_gives_a_different_schedule() -> None:
    assert arrival_schedule(50, 100.0, seed=7) != arrival_schedule(50, 100.0, seed=8)


def test_the_schedule_is_non_decreasing_and_starts_at_zero_or_later() -> None:
    schedule = arrival_schedule(count=200, rate=50.0, seed=1)

    assert schedule[0] >= 0.0
    assert all(b >= a for a, b in pairwise(schedule))


def test_the_mean_inter_arrival_matches_the_requested_rate() -> None:
    """Poisson arrivals: exponential gaps with mean 1/rate. A schedule whose mean
    gap is not 1/rate is not the load the run says it applied."""
    rate = 200.0
    schedule = arrival_schedule(count=20_000, rate=rate, seed=3)

    gaps = [b - a for a, b in pairwise(schedule)]
    mean_gap = sum(gaps) / len(gaps)

    assert abs(mean_gap - 1.0 / rate) < 0.1 / rate


def test_the_gaps_are_exponential_not_uniform() -> None:
    """A uniform schedule is a metronome, and a metronome does not produce
    queues. The distinguishing property: for an exponential, the standard
    deviation equals the mean."""
    rate = 100.0
    schedule = arrival_schedule(count=20_000, rate=rate, seed=5)
    gaps = [b - a for a, b in pairwise(schedule)]

    mean = sum(gaps) / len(gaps)
    variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)

    assert abs(variance**0.5 / mean - 1.0) < 0.05


# ------------------------------------------------------------------ execution


def test_a_closed_loop_issues_every_operation_exactly_once() -> None:
    seen: list[int] = []
    lock = threading.Lock()

    def issue(client: Any, index: int) -> None:
        with lock:
            seen.append(index)

    result = run_load(
        make_client=lambda: object(),
        issue=issue,
        count=200,
        model=LoadModel(clients=8),
    )

    assert sorted(seen) == list(range(200))
    assert len(result.requests) == 200


def test_each_client_gets_its_own_connection() -> None:
    """A shared psycopg connection would serialise the clients and measure the
    lock instead of the database."""
    built = 0
    lock = threading.Lock()

    def make_client() -> object:
        nonlocal built
        with lock:
            built += 1
        return object()

    run_load(make_client, lambda c, i: None, count=64, model=LoadModel(clients=8))

    assert built == 8


def test_a_client_is_closed_even_when_an_operation_raises() -> None:
    """A leaked connection outlives the run and poisons the next one."""
    closed: list[object] = []

    def issue(client: Any, index: int) -> None:
        raise RuntimeError("boom")

    result = run_load(
        lambda: object(),
        issue,
        count=10,
        model=LoadModel(clients=2),
        close_client=closed.append,
    )

    assert len(closed) == 2
    assert result.errors == 10


def test_a_failed_operation_is_recorded_not_dropped() -> None:
    """An error rate that silently shrinks the sample turns a broken system into
    a fast one."""

    def issue(client: Any, index: int) -> None:
        if index % 2 == 0:
            raise RuntimeError("boom")

    result = run_load(lambda: object(), issue, count=20, model=LoadModel(clients=4))

    assert result.errors == 10
    assert result.successes == 10
    assert len(result.requests) == 20


def test_every_request_carries_both_clocks() -> None:
    result = run_load(lambda: object(), lambda c, i: None, count=20, model=LoadModel(clients=4))

    for request in result.requests:
        assert request.response_seconds >= request.service_seconds >= 0.0


def test_response_time_equals_service_time_in_a_closed_loop() -> None:
    """With no schedule there is nothing to be late for, so the two clocks
    coincide — and saying so is what keeps a closed-loop number from being read
    as if it had corrected for queueing."""
    result = run_load(lambda: object(), lambda c, i: None, count=40, model=LoadModel(clients=4))

    assert all(r.scheduled_seconds is None for r in result.requests)
    assert all(r.response_seconds == pytest.approx(r.service_seconds) for r in result.requests)


# ------------------------------------- the property the whole module exists for


def test_open_loop_records_the_queueing_a_closed_loop_would_hide() -> None:
    """The coordinated-omission case, made to happen on purpose.

    One client, requests scheduled every 10 ms, and a server that takes 50 ms.
    A closed loop would simply issue fewer requests and report 50 ms. Measured
    from the schedule, the queue is visible and grows without bound.
    """

    def slow(client: Any, index: int) -> None:
        time.sleep(0.05)

    result = run_load(
        make_client=lambda: object(),
        issue=slow,
        count=8,
        model=LoadModel(clients=1, arrival_rate=100.0),
        schedule=[i * 0.01 for i in range(8)],
    )

    services = [r.service_seconds for r in result.requests]
    responses = [r.response_seconds for r in result.requests]

    # Every request took about the same to serve...
    assert max(services) - min(services) < 0.03
    # ...and the last one waited far longer than it was served.
    assert max(responses) > 3 * max(services)
    assert result.queueing_seconds > 0.0


def test_the_summary_separates_the_two_latencies() -> None:
    """A report that carried one number could not say which it was."""
    result = run_load(
        lambda: object(),
        lambda c, i: time.sleep(0.002),
        count=40,
        model=LoadModel(clients=4, arrival_rate=1000.0),
    )

    summary = summarise_load(result)

    assert "response_p99_ms" in summary
    assert "service_p99_ms" in summary
    assert summary["response_p99_ms"] >= summary["service_p99_ms"]
    assert summary["clients"] == 4
    assert summary["arrival_rate_per_second"] == 1000.0


def test_throughput_is_completed_over_the_measured_window() -> None:
    result = run_load(
        lambda: object(),
        lambda c, i: time.sleep(0.001),
        count=100,
        model=LoadModel(clients=10),
    )

    assert result.duration_seconds > 0
    assert result.throughput == pytest.approx(result.successes / result.duration_seconds, rel=1e-6)


def test_a_closed_loop_summary_says_the_arrival_rate_is_absent() -> None:
    """Not zero. Zero is a rate; absent is the closed loop having none."""
    result = run_load(lambda: object(), lambda c, i: None, count=10, model=LoadModel(clients=2))

    assert summarise_load(result)["arrival_rate_per_second"] is None


def test_concurrency_actually_overlaps() -> None:
    """The check that the clients are not accidentally serialised: with eight
    clients and a 50 ms operation, eight operations take about 50 ms, not 400."""
    started = time.perf_counter()
    run_load(
        lambda: object(),
        lambda c, i: time.sleep(0.05),
        count=8,
        model=LoadModel(clients=8),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"eight concurrent 50ms operations took {elapsed:.3f}s"


# ------------------------------- errors that end a run vs errors that are data
#
# A query that fails is a data point. A system that is *gone* is the end of the
# run: continuing would record a dead database as a stream of query errors and
# let the "did it crash" check pass while it lay dead. The load engine has to
# know the difference, because only the caller does.


class _GoneError(Exception):
    pass


def test_a_fatal_error_stops_the_load_and_propagates() -> None:
    def issue(client: Any, index: int) -> None:
        if index == 3:
            raise _GoneError("the system is gone")

    with pytest.raises(_GoneError):
        run_load(
            lambda: object(),
            issue,
            count=1000,
            model=LoadModel(clients=2),
            fatal=(_GoneError,),
        )


def test_a_fatal_error_still_closes_every_client() -> None:
    closed: list[object] = []

    def issue(client: Any, index: int) -> None:
        raise _GoneError("gone")

    with pytest.raises(_GoneError):
        run_load(
            lambda: object(),
            issue,
            count=10,
            model=LoadModel(clients=3),
            fatal=(_GoneError,),
            close_client=closed.append,
        )

    assert len(closed) == 3


def test_an_error_not_declared_fatal_is_counted_as_before() -> None:
    """Only what the caller names as fatal ends the run. Everything else is a
    measurement of a system that answered badly, which is a finding."""

    def issue(client: Any, index: int) -> None:
        raise RuntimeError("just a bad query")

    result = run_load(
        lambda: object(), issue, count=12, model=LoadModel(clients=2), fatal=(_GoneError,)
    )

    assert result.errors == 12


def test_clients_are_disposed_through_the_closer_the_caller_gave() -> None:
    """The engine does not guess how to dispose of something it did not create:
    an adapter is stopped, a connection is closed, and only the caller knows."""
    stopped: list[int] = []

    result = run_load(
        make_client=lambda: len(stopped),
        issue=lambda c, i: None,
        count=8,
        model=LoadModel(clients=4),
        close_client=lambda c: stopped.append(1),
    )

    assert len(stopped) == 4
    assert result.successes == 8
