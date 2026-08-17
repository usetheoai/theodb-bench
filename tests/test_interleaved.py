"""Measuring two systems query by query instead of one run after the other.

The paired test removes the variance of query difficulty: both systems answered
the same query. It does not remove drift in the machine, because two sequential
runs happen minutes apart. Measured 2026-08-17, the same configuration re-run on
the same host varied by 24% and 46% in median throughput, so a busier host during
one side of a pair is attributed to the engine with the same confidence a real
difference would be — and the confidence interval does not protect against it,
since it measures dispersion across queries rather than across runs.

Interleaving closes that: query *i* goes to both systems back to back, so anything
that changes on the scale of minutes affects both sides equally.

The order alternates per query, and that is not a detail. If system A always goes
first it pays the cold-cache cost of every query and B answers each one with the
page cache just warmed by A — a bias that would look exactly like B being faster.
"""

from __future__ import annotations

from typing import Any

import pytest
from theodb_bench.errors import ConfigError
from theodb_bench.interleaved import InterleavedResult, interleave


class _Recorder:
    """An adapter double that records the order it was asked in."""

    def __init__(self, name: str, latency_ms: float, log: list[str]) -> None:
        self.system_id = name
        self._latency = latency_ms
        self._log = log
        self.seen: list[int] = []

    def execute(self, query: Any) -> Any:
        self._log.append(self.system_id)
        self.seen.append(query)

        class _Result:
            latency_seconds = self._latency / 1000.0
            ids = (1, 2, 3)

        return _Result()


def test_each_query_goes_to_both_systems_before_the_next_one() -> None:
    log: list[str] = []
    a, b = _Recorder("a", 1.0, log), _Recorder("b", 2.0, log)

    interleave(("a", a), ("b", b), queries=[0, 1, 2])

    # a,b then b,a then a,b -- never a,a,a then b,b,b
    assert log[:2] in (["a", "b"], ["b", "a"])
    assert log[2:4] in (["a", "b"], ["b", "a"])
    assert sorted(log) == ["a", "a", "a", "b", "b", "b"]


def test_the_order_alternates_so_neither_side_always_pays_the_cold_cache() -> None:
    """A fixed order would let the second system answer every query with the page
    cache the first just warmed, which looks exactly like being faster."""
    log: list[str] = []
    a, b = _Recorder("a", 1.0, log), _Recorder("b", 1.0, log)

    interleave(("a", a), ("b", b), queries=list(range(6)))

    firsts = [log[i * 2] for i in range(6)]
    assert firsts.count("a") == 3
    assert firsts.count("b") == 3
    assert firsts[0] != firsts[1], "the order must alternate, not be random"


def test_both_sides_answer_the_same_queries_in_the_same_order() -> None:
    log: list[str] = []
    a, b = _Recorder("a", 1.0, log), _Recorder("b", 2.0, log)

    result = interleave(("a", a), ("b", b), queries=[10, 11, 12])

    assert a.seen == b.seen == [10, 11, 12]
    assert isinstance(result, InterleavedResult)
    assert sorted(result.latency_a) == sorted(result.latency_b) == [0, 1, 2]


def test_the_result_pairs_straight_into_the_significance_test() -> None:
    log: list[str] = []
    a, b = _Recorder("a", 1.0, log), _Recorder("b", 3.0, log)

    result = interleave(("a", a), ("b", b), queries=list(range(40)))

    assert result.latency_a[0] == pytest.approx(1.0, abs=0.5)
    assert result.latency_b[0] == pytest.approx(3.0, abs=0.5)
    assert result.name_a == "a"
    assert result.name_b == "b"


def test_a_query_that_fails_on_one_side_is_dropped_from_both() -> None:
    """A pair needs both halves. Keeping the surviving half would compare the
    systems on a query only one of them answered."""

    class _Fails(_Recorder):
        def execute(self, query: Any) -> Any:
            if query == 1:
                raise RuntimeError("boom")
            return super().execute(query)

    log: list[str] = []
    a, b = _Recorder("a", 1.0, log), _Fails("b", 2.0, log)

    result = interleave(("a", a), ("b", b), queries=[0, 1, 2])

    assert sorted(result.latency_a) == [0, 2]
    assert sorted(result.latency_b) == [0, 2]
    assert result.dropped == (1,)


def test_an_empty_query_set_is_refused() -> None:
    log: list[str] = []
    with pytest.raises(ConfigError, match="no queries"):
        interleave(("a", _Recorder("a", 1.0, log)), ("b", _Recorder("b", 1.0, log)), queries=[])
