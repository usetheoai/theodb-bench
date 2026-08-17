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
from theodb_bench.bench.vector import VectorWorkload


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
