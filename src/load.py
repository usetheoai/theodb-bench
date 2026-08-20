"""Issuing work the way a real client population does.

Until this module the harness measured one client at a time — `for i in
range(sample): adapter.execute(...)`. That is a closed loop with a single client,
and it answers a question nobody asks of a database: how fast is it when nothing
else is happening. Throughput under contention is where a database lives, and no
number the harness produced could describe it.

Two things had to exist, and the second is what makes the measurement honest.

**Concurrency.** N clients, each with its own connection, because a psycopg
connection is not safe to share and sharing one would measure the lock rather
than the database. Threads rather than asyncio: the driver releases the GIL on
network I/O, so threads are what the already-installed dependency supports, and
an async driver would be a second dependency for the same result.

**An arrival rate, and latency measured from it.** In a closed loop a stalled
system simply receives fewer requests, so the stall never enters the latency
distribution: the system looks fine and the queue is invisible. That is
coordinated omission (Tene, *How NOT to Measure Latency*). The correction is to
schedule each request in advance and measure from the moment it *should* have
been issued rather than the moment it was.

Every request therefore carries two numbers, and they are never conflated:

    response time = finished - scheduled   (includes queueing; the honest one)
    service  time = finished - started     (what the server took)

Reporting only the service time is the omission. Reporting both makes the queue
visible, and their difference *is* the queueing delay.

Arrivals are Poisson — exponential gaps — rather than a fixed interval. A
metronome does not produce queues, and a load model that cannot produce a queue
cannot measure what happens in one. The schedule is seeded so that two runs of
the same benchmark differ only in the system under test.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

Client = TypeVar("Client")

#: Default seed for the arrival process. Fixed on purpose: an unseeded schedule
#: would make two runs differ in their load as well as in the system.
DEFAULT_LOAD_SEED = 20260818


@dataclass(frozen=True)
class LoadModel:
    """How requests are issued: how many clients, and at what rate."""

    clients: int = 1
    arrival_rate: float | None = None
    """Requests per second, across all clients. `None` is a closed loop: each
    client issues its next request when the previous one returns."""
    seed: int = DEFAULT_LOAD_SEED

    def __post_init__(self) -> None:
        if self.clients < 1:
            raise ValueError(f"a load needs at least 1 client, got {self.clients}")
        if self.arrival_rate is not None and self.arrival_rate <= 0:
            raise ValueError(
                f"arrival_rate must be positive, got {self.arrival_rate}. A rate of zero "
                f"schedules the first request at infinity; omit it for a closed loop."
            )

    @property
    def is_closed_loop(self) -> bool:
        return self.arrival_rate is None


@dataclass(frozen=True)
class Request:
    """One issued operation, with both clocks."""

    index: int
    client: int
    ok: bool
    service_seconds: float
    response_seconds: float
    scheduled_seconds: float | None = None
    error: str | None = None

    @property
    def queueing_seconds(self) -> float:
        """How long it waited past its schedule. Zero in a closed loop."""
        return max(0.0, self.response_seconds - self.service_seconds)


@dataclass
class LoadResult:
    """Everything one load run observed."""

    requests: list[Request] = field(default_factory=list)
    duration_seconds: float = 0.0
    model: LoadModel = field(default_factory=LoadModel)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.requests if r.ok)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.requests if not r.ok)

    @property
    def throughput(self) -> float:
        return self.successes / self.duration_seconds if self.duration_seconds > 0 else 0.0

    @property
    def queueing_seconds(self) -> float:
        """Total time requests spent waiting past their schedule."""
        return sum(r.queueing_seconds for r in self.requests)


def arrival_schedule(count: int, rate: float, seed: int = DEFAULT_LOAD_SEED) -> list[float]:
    """Poisson arrival times, in seconds from the start of the measured window.

    Exponential gaps with mean `1/rate`. A fixed interval is a metronome and a
    metronome does not build queues; the whole reason to schedule arrivals is to
    let a queue form when the system cannot keep up.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")

    # Deliberately not a cryptographic generator: what this needs is
    # reproducibility, so that two runs of the same benchmark apply the same
    # load. `secrets` would give the opposite property.
    rng = random.Random(seed)  # noqa: S311
    schedule: list[float] = []
    at = 0.0
    for _ in range(count):
        at += rng.expovariate(rate)
        schedule.append(at)
    return schedule


def run_load(
    make_client: Callable[[], Client],
    issue: Callable[[Client, int], Any],
    count: int,
    model: LoadModel,
    schedule: Sequence[float] | None = None,
    fatal: tuple[type[BaseException], ...] = (),
    close_client: Callable[[Client], None] | None = None,
) -> LoadResult:
    """Issue `count` operations under `model`, recording both clocks per request.

    `make_client` is called once per client, so each gets its own connection.
    `issue(client, index)` performs one operation and raises on failure; a
    failure is recorded rather than dropped, because an error rate that quietly
    shrinks the sample turns a broken system into a fast one.

    `fatal` names the exceptions that end the load instead of becoming data. A
    query that fails is a measurement of a system answering badly; a system that
    is *gone* is the end of the run, and counting it as a stream of query errors
    would let the "did it crash" check pass while it lay dead. Only the caller
    knows which is which, so the engine is told rather than guessing.

    `close_client` disposes of each client. The engine does not guess how: an
    adapter is stopped, a connection is closed, and a caller that passes nothing
    is saying there is nothing to dispose of.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")

    rate = model.arrival_rate
    if schedule is None and rate is not None:
        schedule = arrival_schedule(count, rate, model.seed)

    result = LoadResult(model=model)
    lock = threading.Lock()
    next_index = _Counter()

    clients: list[Client] = []
    stop = threading.Event()
    fatal_error: list[BaseException] = []
    started_at = 0.0

    def worker(client_id: int) -> None:
        client = make_client()
        with lock:
            clients.append(client)
        while not stop.is_set():
            index = next_index.take(count)
            if index is None:
                return
            try:
                _issue_one(
                    client, client_id, index, issue, schedule, started_at, result, lock, fatal
                )
            except BaseException as exc:  # only `fatal` reaches here
                with lock:
                    fatal_error.append(exc)
                stop.set()
                return

    try:
        with ThreadPoolExecutor(max_workers=model.clients) as pool:
            started_at = time.perf_counter()
            list(pool.map(worker, range(model.clients)))
            result.duration_seconds = time.perf_counter() - started_at
    finally:
        # Disposed even when the load ended fatally: a leaked connection
        # outlives the run and poisons the next.
        if close_client is not None:
            for client in clients:
                close_client(client)

    if fatal_error:
        raise fatal_error[0]

    result.requests.sort(key=lambda r: r.index)
    return result


def _issue_one(
    client: Client,
    client_id: int,
    index: int,
    issue: Callable[[Client, int], Any],
    schedule: Sequence[float] | None,
    started_at: float,
    result: LoadResult,
    lock: threading.Lock,
    fatal: tuple[type[BaseException], ...] = (),
) -> None:
    """Issue one operation, waiting for its slot when the load is open-loop."""
    scheduled: float | None = None
    if schedule is not None and index < len(schedule):
        scheduled = schedule[index]
        # Sleep only until the slot; being *late* is the measurement, not an
        # error to correct for, so a missed slot is issued immediately and its
        # lateness lands in the response time.
        delay = (started_at + scheduled) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    send = time.perf_counter()
    error: str | None = None
    try:
        issue(client, index)
    except fatal:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    done = time.perf_counter()

    service = done - send
    response = (done - (started_at + scheduled)) if scheduled is not None else service

    with lock:
        result.requests.append(
            Request(
                index=index,
                client=client_id,
                ok=error is None,
                service_seconds=service,
                response_seconds=max(response, service),
                scheduled_seconds=scheduled,
                error=error,
            )
        )


class _Counter:
    """Hands each index to exactly one client."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def take(self, limit: int) -> int | None:
        with self._lock:
            if self._value >= limit:
                return None
            index = self._value
            self._value += 1
            return index


def summarise_load(result: LoadResult) -> dict[str, Any]:
    """The load's shape, with the two latencies kept apart.

    A summary carrying one latency could not say which it was, and the whole
    point of scheduling arrivals is that the two differ.
    """
    responses = sorted(r.response_seconds * 1000.0 for r in result.requests if r.ok)
    services = sorted(r.service_seconds * 1000.0 for r in result.requests if r.ok)

    return {
        "clients": result.model.clients,
        # Absent, not zero: a closed loop has no arrival rate, and zero is a rate.
        "arrival_rate_per_second": result.model.arrival_rate,
        "loop": "closed" if result.model.is_closed_loop else "open",
        "successes": result.successes,
        "errors": result.errors,
        "duration_seconds": result.duration_seconds,
        "throughput_per_second": result.throughput,
        "response_p50_ms": _percentile(responses, 50),
        "response_p95_ms": _percentile(responses, 95),
        "response_p99_ms": _percentile(responses, 99),
        "service_p50_ms": _percentile(services, 50),
        "service_p95_ms": _percentile(services, 95),
        "service_p99_ms": _percentile(services, 99),
        "queueing_seconds_total": result.queueing_seconds,
    }


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Nearest-rank percentile; `None` when nothing succeeded."""
    if not sorted_values:
        return None
    rank = max(1, round(percentile / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]
