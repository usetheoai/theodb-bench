"""The orchestrator depends on a protocol, not on one workload family.

Measured 2026-08-17: `RunRequest.workload` was typed `VectorWorkload` and the
runner constructed `VectorBenchmark` directly, so the analytical surface — 336
lines of `AnalyticalBenchmark` with its own oracle, the adapter methods, and the
four-state residency gate — could not be invoked by `theodb-bench run` at all.
`bench/analytical.py`, `bench/graph.py` and `bench/retrieval.py` had zero
importers in `src/`.

Four sites carried the coupling, and each one asked a question a workload family
should answer for itself: which benchmark to build, what the benchmark artifact
says, how many operations are expected, and how many were warm-up. They are now
protocol members, so a second family is a module rather than a change to the
orchestrator (`rules/architecture.md § 2`).
"""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import IndexSpec
from theodb_bench.bench.analytical import AnalyticalWorkload
from theodb_bench.bench.protocol import Benchmark, Workload
from theodb_bench.bench.vector import VectorWorkload, summarise_latency


@pytest.mark.parametrize(
    "workload",
    [
        VectorWorkload(
            corpus_size=100,
            dimension=4,
            query_count=10,
            k=5,
            warmup_queries=2,
            indexes=(IndexSpec(kind="none"),),
        ),
        AnalyticalWorkload(row_count=100),
    ],
    ids=["vector", "analytical"],
)
def test_every_workload_family_satisfies_the_protocol(workload: object) -> None:
    assert isinstance(workload, Workload)


def test_a_vector_workload_reports_its_own_operation_count() -> None:
    workload = VectorWorkload(
        corpus_size=100,
        dimension=4,
        query_count=10,
        k=5,
        warmup_queries=2,
        indexes=(IndexSpec(kind="none"),),
    )

    assert workload.expected_operations(measured_points=3, repetitions=2) == 3 * 2 * 10
    assert workload.warmup_operations == 2
    assert workload.benchmark_payload()["workload"]["type"] == "ann"


def test_an_analytical_workload_reports_its_own_operation_count() -> None:
    """Its unit is a query per path per repetition, not a k-NN probe."""
    workload = AnalyticalWorkload(row_count=100, repetitions=3)

    payload = workload.benchmark_payload()

    assert payload["workload"]["type"] == "analytical"
    assert payload["workload"]["operation_count"] == len(workload.queries) * len(workload.paths)
    assert payload["quality"]["metric"] == "exact_match"
    assert "k" not in payload["workload"], "an aggregation has no k"
    assert workload.warmup_operations == workload.warmup_queries


def test_a_vector_workload_builds_a_vector_benchmark() -> None:
    import numpy as np

    workload = VectorWorkload(
        corpus_size=8,
        dimension=4,
        query_count=2,
        k=2,
        warmup_queries=0,
        indexes=(IndexSpec(kind="none"),),
    )
    corpus = np.zeros((8, 4), dtype=np.float32)
    queries = np.zeros((2, 4), dtype=np.float32)

    benchmark = workload.build(corpus=corpus, queries=queries)

    assert isinstance(benchmark, Benchmark)


def test_an_analytical_workload_builds_an_analytical_benchmark() -> None:
    """It generates its own rows from its seed, so it needs no corpus passed in."""
    benchmark = AnalyticalWorkload(row_count=50).build(corpus=None, queries=None)

    assert isinstance(benchmark, Benchmark)


# ------------------------------------- the frontier a swept run is supposed to emit
#
# Measured 2026-08-17: `analysis/pareto.py` implemented dominance and frontier
# computation and had zero importers in `src/`. The project's own rule says a
# headline QPS comparison requires a stated target recall with its interpolation
# method **or the complete Pareto frontier** — and no run emitted one.


def test_a_swept_run_emits_its_pareto_frontier() -> None:
    from theodb_bench.bench.vector import PointResult, RepetitionResult
    from theodb_bench.report import pareto_payload

    def _point(label: str, qps: float, recall: float) -> PointResult:
        point = PointResult(label=label, parameters={})
        point.repetitions.append(
            RepetitionResult(
                repetition=1,
                successes=10,
                errors=0,
                timeouts=0,
                duration_seconds=10 / qps,
                latency=summarise_latency([1.0] * 10),
                recall=recall,
            )
        )
        return point

    points = [
        _point("shallow", 600.0, 0.70),   # fast, poor quality -- on the frontier
        _point("middle", 400.0, 0.95),    # on the frontier
        _point("dominated", 300.0, 0.90), # slower AND worse than middle
        _point("deep", 150.0, 0.99),      # on the frontier
    ]

    payload = pareto_payload(points)

    assert set(payload["frontier"]) == {"shallow", "middle", "deep"}
    dominated = [p for p in payload["points"] if p["label"] == "dominated"]
    assert dominated and dominated[0]["dominated_by"] == ["middle"]


def test_a_run_with_one_configuration_emits_no_frontier() -> None:
    """A frontier of one point is a point. Emitting it would dress a single
    measurement as a trade-off curve."""
    from theodb_bench.bench.vector import PointResult
    from theodb_bench.report import pareto_payload

    assert pareto_payload([PointResult(label="only", parameters={})]) is None


# ------------------------------------------- regression detection has a caller
#
# Measured 2026-08-17: `analysis/regression.py` implemented comparability checks,
# gates and a verdict, and had zero importers in `src/`. Three profiles declared
# `regression_gate = True` while nothing compared anything to a baseline.


def test_a_regression_against_an_incomparable_baseline_fails_closed() -> None:
    from theodb_bench.regression_wiring import regression_for

    payload, comparable = regression_for(
        candidate={
            "run_id": "cand",
            "benchmark_id": "vector/synthetic/sweep",
            "benchmark_version": 1,
            "profile": "pr",
            "system": "theodb",
            "hardware_class": "droplet",
            "metrics": {"throughput_per_second": 400.0},
        },
        baseline={
            "run_id": "base",
            "benchmark_id": "vector/synthetic/smoke",   # a different benchmark
            "benchmark_version": 1,
            "profile": "pr",
            "system": "theodb",
            "hardware_class": "droplet",
            "metrics": {"throughput_per_second": 500.0},
        },
    )

    assert comparable is False
    assert payload["verdict"] == "INCOMPARABLE"


def test_a_comparable_baseline_produces_a_verdict() -> None:
    from theodb_bench.regression_wiring import regression_for

    common = {
        "benchmark_id": "vector/synthetic/sweep",
        "benchmark_version": 1,
        "profile": "pr",
        "system": "theodb",
        "hardware_class": "droplet",
    }
    payload, comparable = regression_for(
        candidate={**common, "run_id": "cand", "metrics": {"throughput_per_second": 400.0}},
        baseline={**common, "run_id": "base", "metrics": {"throughput_per_second": 500.0}},
    )

    assert comparable is True
    assert payload["verdict"] != "INCOMPARABLE"
    assert payload["baseline"]["run_id"] == "base"
