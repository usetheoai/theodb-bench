"""A dominated point must never appear on the frontier."""

from __future__ import annotations

import pytest
from theodb_bench.analysis.pareto import (
    MAXIMIZE,
    MINIMIZE,
    Objective,
    Point,
    dominates,
    frontier,
    matched_quality,
    pareto_payload,
)
from theodb_bench.schemas import validate

QUALITY = Objective("recall_at_10", MAXIMIZE)
THROUGHPUT = Objective("qps", MAXIMIZE)
LATENCY = Objective("p99_ms", MINIMIZE)
MEMORY = Objective("rss_bytes", MINIMIZE)


def _point(label: str, **values: float) -> Point:
    return Point(label=label, values=values)


# ----------------------------------------------------------------- domination


def test_a_strictly_better_point_dominates() -> None:
    better = _point("a", recall_at_10=0.99, qps=500.0)
    worse = _point("b", recall_at_10=0.90, qps=400.0)
    assert dominates(better, worse, [QUALITY, THROUGHPUT])
    assert not dominates(worse, better, [QUALITY, THROUGHPUT])


def test_a_tie_on_every_objective_dominates_nothing() -> None:
    left = _point("a", recall_at_10=0.9, qps=100.0)
    right = _point("b", recall_at_10=0.9, qps=100.0)
    assert not dominates(left, right, [QUALITY, THROUGHPUT])
    assert not dominates(right, left, [QUALITY, THROUGHPUT])


def test_equal_on_one_and_better_on_another_dominates() -> None:
    left = _point("a", recall_at_10=0.9, qps=200.0)
    right = _point("b", recall_at_10=0.9, qps=100.0)
    assert dominates(left, right, [QUALITY, THROUGHPUT])


def test_a_trade_off_dominates_nothing() -> None:
    fast = _point("fast", recall_at_10=0.80, qps=900.0)
    accurate = _point("accurate", recall_at_10=0.99, qps=200.0)
    assert not dominates(fast, accurate, [QUALITY, THROUGHPUT])
    assert not dominates(accurate, fast, [QUALITY, THROUGHPUT])


def test_a_missing_value_prevents_domination_rather_than_assuming_one() -> None:
    complete = _point("a", recall_at_10=0.99, qps=500.0)
    partial = _point("b", recall_at_10=0.90)
    assert not dominates(complete, partial, [QUALITY, THROUGHPUT])


def test_minimised_objectives_invert_the_comparison() -> None:
    quick = _point("a", p99_ms=2.0)
    slow = _point("b", p99_ms=9.0)
    assert dominates(quick, slow, [LATENCY])


# ------------------------------------------------------------------- frontier


def test_the_frontier_excludes_dominated_points() -> None:
    points = [
        _point("fast", recall_at_10=0.80, qps=900.0),
        _point("balanced", recall_at_10=0.95, qps=500.0),
        _point("accurate", recall_at_10=0.99, qps=200.0),
        _point("pointless", recall_at_10=0.70, qps=300.0),
    ]
    labels = [p.label for p in frontier(points, [QUALITY, THROUGHPUT])]
    assert labels == ["fast", "balanced", "accurate"]
    assert "pointless" not in labels


def test_an_empty_set_has_an_empty_frontier() -> None:
    assert frontier([], [QUALITY]) == []


def test_a_single_point_is_its_own_frontier() -> None:
    only = _point("only", recall_at_10=0.5, qps=1.0)
    assert [p.label for p in frontier([only], [QUALITY, THROUGHPUT])] == ["only"]


def test_identical_points_both_survive() -> None:
    # Neither dominates the other, so dropping one would be an arbitrary choice.
    points = [_point("a", qps=100.0), _point("b", qps=100.0)]
    assert len(frontier(points, [THROUGHPUT])) == 2


def test_four_objectives_are_handled_together() -> None:
    points = [
        _point("a", recall_at_10=0.99, qps=500.0, p99_ms=2.0, rss_bytes=1e9),
        _point("b", recall_at_10=0.98, qps=400.0, p99_ms=3.0, rss_bytes=2e9),
    ]
    labels = [p.label for p in frontier(points, [QUALITY, THROUGHPUT, LATENCY, MEMORY])]
    assert labels == ["a"]


def test_a_frontier_needs_an_objective() -> None:
    with pytest.raises(ValueError, match="at least one objective"):
        frontier([_point("a", qps=1.0)], [])


def test_an_unknown_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        Objective("qps", "sideways")


# ------------------------------------------------------------ matched quality


def test_matched_quality_picks_the_fastest_configuration_that_reaches_the_target() -> None:
    points = [
        _point("ef16", recall_at_10=0.90, qps=900.0),
        _point("ef64", recall_at_10=0.96, qps=500.0),
        _point("ef256", recall_at_10=0.99, qps=200.0),
    ]
    matched = matched_quality(points, "recall_at_10", 0.95, "qps")
    assert matched.selected == "ef64"
    assert matched.method == "nearest_at_or_above"


def test_matched_quality_refuses_to_promote_a_near_miss() -> None:
    # 0.949 is not 0.95. Reporting it as if it were is how a matched-recall
    # comparison stops being matched.
    points = [_point("ef16", recall_at_10=0.949, qps=900.0)]
    matched = matched_quality(points, "recall_at_10", 0.95, "qps")
    assert matched.selected is None
    assert matched.method == "none_available"
    assert "0.949" in matched.detail


def test_matched_quality_reports_the_best_observed_when_none_qualifies() -> None:
    points = [_point("a", recall_at_10=0.5, qps=10.0), _point("b", recall_at_10=0.7, qps=5.0)]
    assert "0.7" in matched_quality(points, "recall_at_10", 0.99, "qps").detail


# -------------------------------------------------------------------- payload


def test_payload_validates_and_marks_dominated_points() -> None:
    points = [
        _point("fast", recall_at_10=0.80, qps=900.0),
        _point("accurate", recall_at_10=0.99, qps=200.0),
        _point("dominated", recall_at_10=0.70, qps=100.0),
    ]
    payload = pareto_payload(
        points,
        [QUALITY, THROUGHPUT],
        run_id="20260813T000000Z-vector-fake-abcdef",
        matched=matched_quality(points, "recall_at_10", 0.95, "qps"),
    )
    validate("pareto", payload)
    assert payload["frontier"] == ["fast", "accurate"]
    dominated = next(p for p in payload["points"] if p["label"] == "dominated")
    assert dominated["dominated_by"]
