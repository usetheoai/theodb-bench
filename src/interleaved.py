"""Measuring two systems query by query, so machine drift cancels instead of counting.

A paired test over two sequential runs removes the variance of query difficulty —
both systems answered the same query — and leaves the variance of the machine,
because the runs happened minutes apart. Measured on 2026-08-17, the same
configuration re-run on the same host varied by 24% and 46% in median throughput.
A busier host during one side of a pair is then attributed to the engine with the
same confidence a real difference would be, and the confidence interval offers no
protection: it measures dispersion across queries, not across runs.

Interleaving sends query *i* to both systems back to back, so anything moving on
the scale of minutes moves both sides together.

The order alternates per query, and that is the load-bearing part. With a fixed
order the first system pays the cold-cache cost of every query and the second
answers each one with the page cache just warmed — a bias indistinguishable from
the second system being faster. Alternating makes each side pay half.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from theodb_bench.errors import ConfigError, ErrorContext, Phase


class _Executable(Protocol):
    """The part of a system adapter this needs: answer a query, report how long."""

    system_id: str

    def execute(self, query: Any) -> Any: ...


@dataclass(frozen=True)
class InterleavedResult:
    """Two systems' latencies for the same queries, taken back to back."""

    name_a: str
    name_b: str
    latency_a: dict[int, float]
    latency_b: dict[int, float]

    #: Ids each system returned, per query. Kept so the caller can score quality
    #: on exactly the queries that formed pairs: a latency verdict between two
    #: operating points of different quality compares the operating points, not
    #: the systems.
    returned_a: dict[int, tuple[int, ...]] = field(default_factory=dict)
    returned_b: dict[int, tuple[int, ...]] = field(default_factory=dict)

    #: Query indices dropped because one side failed to answer them. A pair needs
    #: both halves; keeping the surviving half would compare the systems on a
    #: query only one of them answered.
    dropped: tuple[int, ...] = field(default_factory=tuple)

    #: Recorded so a report can say which confounder is in play rather than
    #: leaving a reader to assume the stronger protection.
    interleaved: bool = True


def interleave(
    system_a: tuple[str, _Executable],
    system_b: tuple[str, _Executable],
    *,
    queries: Sequence[Any],
) -> InterleavedResult:
    """Run every query on both systems, alternating which goes first.

    Both systems must already be loaded with the same corpus and built at the
    configurations being compared — this measures, it does not set up. The result
    pairs straight into `compare.pair_by_query`.
    """
    if not queries:
        raise ConfigError(
            "no queries to interleave: an interleaved comparison needs a query set "
            "both systems answer",
            context=ErrorContext(phase=Phase.MEASUREMENT),
        )

    name_a, adapter_a = system_a
    name_b, adapter_b = system_b
    latency_a: dict[int, float] = {}
    latency_b: dict[int, float] = {}
    returned_a: dict[int, tuple[int, ...]] = {}
    returned_b: dict[int, tuple[int, ...]] = {}
    dropped: list[int] = []

    for index, query in enumerate(queries):
        # Even queries lead with A, odd with B. Deterministic rather than random
        # so a run reproduces, and balanced so neither side always pays the cold
        # cache.
        side_a = (name_a, adapter_a, latency_a, returned_a)
        side_b = (name_b, adapter_b, latency_b, returned_b)
        order = (side_a, side_b) if index % 2 == 0 else (side_b, side_a)

        measured: list[tuple[dict[int, float], float, dict[int, tuple[int, ...]], Any]] = []
        for _, adapter, sink, ids_sink in order:
            try:
                result = adapter.execute(query)
            except Exception:  # any failure drops the pair, see below
                measured.clear()
                break
            measured.append(
                (sink, float(result.latency_seconds) * 1000.0, ids_sink, tuple(result.ids))
            )

        if len(measured) != 2:
            # One side did not answer. The pair is incomplete, and half a pair is
            # not a datum: recording it would let the comparison rest on queries
            # only one system could answer, which favours whichever system failed
            # on its hardest ones.
            dropped.append(index)
            continue

        for sink, elapsed, ids_sink, ids in measured:
            sink[index] = elapsed
            ids_sink[index] = ids

    return InterleavedResult(
        name_a=name_a,
        name_b=name_b,
        latency_a=latency_a,
        latency_b=latency_b,
        returned_a=returned_a,
        returned_b=returned_b,
        dropped=tuple(dropped),
    )
