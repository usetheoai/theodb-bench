"""The graph workload: traversal, fanout sweeps, and build cost.

The rule this module exists to enforce: **a traversal result is validated
before its timing is accepted**. A fast traversal that returns the wrong
neighbourhood is not a fast traversal, and a benchmark that timed the walk
without checking the answer would rank a broken implementation first.

Validation is exact rather than sampled: the benchmark computes the true k-hop
neighbourhood itself, from the same edge list the system was given, and refuses
the measurement when they disagree.

The primary unit is **work**, not answer size. `edges_visited` and `ns/edge`
describe what the traversal cost; a query returning few vertices after walking
many edges is expensive, and reporting only the result size hides that.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from theodb_bench.adapters.base import GraphSpec, SystemAdapter, TraversalQuery
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.errors import ConfigError, ErrorContext, Phase

DEFAULT_GRAPH: Final[str] = "bench_graph"

ONE_HOP: Final[str] = "1_hop"
TWO_HOP: Final[str] = "2_hop"
THREE_HOP: Final[str] = "3_hop"
BFS: Final[str] = "bfs"
FANOUT_SWEEP: Final[str] = "fanout_sweep"
BUILD: Final[str] = "build"
REBUILD: Final[str] = "rebuild"
NEIGHBOURHOOD: Final[str] = "graphrag_neighbourhood"

WORKLOADS: Final[tuple[str, ...]] = (
    ONE_HOP,
    TWO_HOP,
    THREE_HOP,
    BFS,
    FANOUT_SWEEP,
    BUILD,
    REBUILD,
    NEIGHBOURHOOD,
)

_HOPS: Final[dict[str, int]] = {ONE_HOP: 1, TWO_HOP: 2, THREE_HOP: 3, BFS: 4}


@dataclass(frozen=True)
class GraphWorkload:
    """A declarative graph workload."""

    vertex_count: int
    average_degree: int = 8
    query_count: int = 100
    seed: int = 20260813
    graph: str = DEFAULT_GRAPH
    directed: bool = True
    workloads: tuple[str, ...] = WORKLOADS
    fanout_degrees: tuple[int, ...] = (2, 8, 32)
    neighbourhood_limit: int = 50
    """Cap for the GraphRAG-style expansion, which is bounded by design."""

    def __post_init__(self) -> None:
        unknown = set(self.workloads) - set(WORKLOADS)
        if unknown:
            raise ConfigError(
                f"unknown graph workload(s): {', '.join(sorted(unknown))}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.vertex_count < 2:
            raise ConfigError(
                "a graph needs at least 2 vertices",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.average_degree < 1:
            raise ConfigError(
                "average degree must be at least 1",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )

    def spec(self) -> GraphSpec:
        return GraphSpec(name=self.graph, directed=self.directed)


def generate_graph(workload: GraphWorkload, degree: int | None = None) -> list[tuple[int, int]]:
    """A seeded edge list.

    Seeded so the same workload builds a bit-identical graph, which is what
    lets two runs differ only in the system under test.
    """
    rng = np.random.default_rng(workload.seed)
    out_degree = degree if degree is not None else workload.average_degree
    edges: list[tuple[int, int]] = []
    for source in range(workload.vertex_count):
        targets = rng.integers(0, workload.vertex_count, size=out_degree)
        edges.extend((source, int(target)) for target in targets if int(target) != source)
    return edges


def build_adjacency(
    edges: Sequence[tuple[int, int]], vertex_count: int, *, directed: bool
) -> dict[int, list[int]]:
    """The reference structure the benchmark validates against."""
    adjacency: dict[int, set[int]] = {v: set() for v in range(vertex_count)}
    for source, target in edges:
        adjacency[source].add(target)
        if not directed:
            adjacency[target].add(source)
    return {vertex: sorted(targets) for vertex, targets in adjacency.items()}


def true_neighbourhood(adjacency: dict[int, list[int]], source: int, hops: int) -> list[int]:
    """The exact k-hop neighbourhood, in discovery order.

    Computed here, from the same edges the system was given. This is the oracle
    a traversal is checked against, and it is why a wrong answer cannot be
    reported as a fast one.
    """
    seen = {source}
    reached: list[int] = []
    frontier: deque[int] = deque([source])
    for _ in range(hops):
        for _ in range(len(frontier)):
            vertex = frontier.popleft()
            for neighbour in adjacency.get(vertex, ()):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                reached.append(neighbour)
                frontier.append(neighbour)
        if not frontier:
            break
    return reached


@dataclass
class GraphResult:
    """One graph workload."""

    workload: str
    status: str = "measured"
    status_detail: str | None = None
    queries: int = 0
    latency: LatencySummary | None = None
    edges_visited: int = 0
    edges_per_second: float | None = None
    nanoseconds_per_edge: float | None = None
    build_seconds: float | None = None
    structure_bytes: int | None = None
    bytes_per_edge: float | None = None
    incorrect_traversals: int = 0
    """Traversals whose result disagreed with the oracle. Any value above zero
    invalidates the timing that accompanied them."""

    fanout: dict[int, float] = field(default_factory=dict)
    """Degree to nanoseconds-per-edge, for the sweep."""

    def metric_series(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        if self.edges_per_second is not None:
            series["edges_per_second"] = [self.edges_per_second]
        if self.nanoseconds_per_edge is not None:
            series["nanoseconds_per_edge"] = [self.nanoseconds_per_edge]
        if self.build_seconds is not None:
            series["build_seconds"] = [self.build_seconds]
        if self.bytes_per_edge is not None:
            series["bytes_per_edge"] = [self.bytes_per_edge]
        if self.latency is not None:
            for name in ("p50", "p95", "p99"):
                value = getattr(self.latency, name)
                if isinstance(value, float):
                    series[f"latency_{name}_ms"] = [value]
        return series

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "status": self.status,
            "status_detail": self.status_detail,
            "queries": self.queries,
            "edges_visited": self.edges_visited,
            "edges_per_second": self.edges_per_second,
            "nanoseconds_per_edge": self.nanoseconds_per_edge,
            "build_seconds": self.build_seconds,
            "structure_bytes": self.structure_bytes,
            "bytes_per_edge": self.bytes_per_edge,
            "incorrect_traversals": self.incorrect_traversals,
            "latency_ms": self.latency.as_dict() if self.latency else None,
            "fanout": {str(degree): value for degree, value in self.fanout.items()},
        }


class GraphBenchmark:
    """Runs graph workloads and validates every traversal before timing it."""

    def __init__(self, workload: GraphWorkload) -> None:
        self.workload = workload
        self.edges = generate_graph(workload)
        self.adjacency = build_adjacency(
            self.edges, workload.vertex_count, directed=workload.directed
        )
        rng = np.random.default_rng(workload.seed + 1)
        self.sources = [
            int(v) for v in rng.integers(0, workload.vertex_count, size=workload.query_count)
        ]

    def load(self, adapter: SystemAdapter) -> GraphResult:
        """Build the structure, timed apart from any query."""
        result = GraphResult(workload=BUILD)
        outcome = adapter.load_graph(self.workload.spec(), self.edges, self.workload.vertex_count)
        result.build_seconds = outcome.seconds
        result.structure_bytes = outcome.index_size_bytes
        stats = adapter.graph_stats()
        per_edge = stats.get("bytes_per_edge")
        result.bytes_per_edge = float(per_edge) if isinstance(per_edge, (int, float)) else None
        result.edges_visited = int(stats.get("edges", 0))
        return result

    def run(self, adapter: SystemAdapter, name: str) -> GraphResult:
        if name not in WORKLOADS:
            raise ConfigError(
                f"unknown graph workload {name!r}", context=ErrorContext(phase=Phase.MEASUREMENT)
            )
        if not adapter.supports("graph"):
            return GraphResult(
                workload=name,
                status="unsupported",
                status_detail=f"{adapter.system_id} has no graph traversal",
            )

        if name in (BUILD, REBUILD):
            return self.load(adapter)
        if name == FANOUT_SWEEP:
            return self._fanout_sweep(adapter)
        if name == NEIGHBOURHOOD:
            return self._traverse(adapter, name, hops=2, limit=self.workload.neighbourhood_limit)
        return self._traverse(adapter, name, hops=_HOPS[name], limit=None)

    # ------------------------------------------------------------- traversal

    def _traverse(
        self, adapter: SystemAdapter, name: str, *, hops: int, limit: int | None
    ) -> GraphResult:
        result = GraphResult(workload=name)
        latencies: list[float] = []
        total_edges = 0
        total_seconds = 0.0

        for source in self.sources:
            outcome = adapter.traverse(
                TraversalQuery(graph=self.workload.graph, source=source, hops=hops, limit=limit)
            )
            if not self._is_correct(source, hops, limit, outcome.vertices):
                # The timing that accompanied a wrong answer is not evidence
                # about traversal speed.
                result.incorrect_traversals += 1
                continue
            latencies.append(outcome.latency_seconds * 1000.0)
            total_edges += outcome.edges_visited
            total_seconds += outcome.latency_seconds

        result.queries = len(latencies)
        result.latency = summarise_latency(latencies)
        result.edges_visited = total_edges
        if total_seconds > 0 and total_edges > 0:
            result.edges_per_second = total_edges / total_seconds
            result.nanoseconds_per_edge = total_seconds * 1e9 / total_edges

        if result.incorrect_traversals:
            result.status = "invalid"
            result.status_detail = (
                f"{result.incorrect_traversals} traversal(s) disagreed with the oracle; "
                "their timings were discarded because a wrong answer is not a fast one"
            )
        return result

    def _is_correct(
        self, source: int, hops: int, limit: int | None, returned: Sequence[int]
    ) -> bool:
        """Compare against the exact neighbourhood computed here.

        With a limit, the system may return any prefix-sized subset of the true
        neighbourhood, so membership and count are checked rather than order --
        a bounded expansion is not required to agree on which vertices it drops.
        """
        expected = true_neighbourhood(self.adjacency, source, hops)
        if limit is None:
            return list(returned) == expected
        expected_set = set(expected)
        return len(returned) == min(limit, len(expected)) and set(returned) <= expected_set

    # ---------------------------------------------------------------- fanout

    def _fanout_sweep(self, adapter: SystemAdapter) -> GraphResult:
        """Cost per edge as out-degree grows.

        The interesting shape is whether ns/edge stays flat. A traversal whose
        per-edge cost rises with degree has a different scaling story from one
        whose total cost simply rises because there are more edges.
        """
        result = GraphResult(workload=FANOUT_SWEEP)
        for degree in self.workload.fanout_degrees:
            edges = generate_graph(self.workload, degree=degree)
            adjacency = build_adjacency(
                edges, self.workload.vertex_count, directed=self.workload.directed
            )
            adapter.load_graph(self.workload.spec(), edges, self.workload.vertex_count)

            total_edges = 0
            total_seconds = 0.0
            for source in self.sources[: min(20, len(self.sources))]:
                outcome = adapter.traverse(
                    TraversalQuery(graph=self.workload.graph, source=source, hops=2)
                )
                if list(outcome.vertices) != true_neighbourhood(adjacency, source, 2):
                    result.incorrect_traversals += 1
                    continue
                total_edges += outcome.edges_visited
                total_seconds += outcome.latency_seconds
            if total_edges:
                result.fanout[degree] = total_seconds * 1e9 / total_edges
                result.edges_visited += total_edges

        # Restore the declared graph so a later workload measures what it declared.
        adapter.load_graph(self.workload.spec(), self.edges, self.workload.vertex_count)
        if result.incorrect_traversals:
            result.status = "invalid"
            result.status_detail = (
                f"{result.incorrect_traversals} traversal(s) disagreed with the oracle"
            )
        return result


def rebuild_delta(first: GraphResult, second: GraphResult) -> dict[str, Any]:
    """Compare an initial build with a rebuild over the same edges."""
    if first.build_seconds is None or second.build_seconds is None:
        return {"delta_seconds": None, "note": "both builds must be measured to compare them"}
    return {
        "first_build_seconds": first.build_seconds,
        "rebuild_seconds": second.build_seconds,
        "delta_seconds": second.build_seconds - first.build_seconds,
        "note": (
            "A rebuild materially faster than the first build usually means "
            "something was reused; a benchmark that ran only one of them would "
            "not know which number it had."
        ),
    }


def timed_reference_traversal(
    adjacency: dict[int, list[int]], sources: Sequence[int], hops: int
) -> tuple[float, int]:
    """Walk the reference structure, for a floor to compare a system against.

    Not a competitor: an in-process dictionary walk with no durability, no
    concurrency and no storage. It exists so that a system's ns/edge can be
    read against something, and the report says exactly what it is.
    """
    started = time.perf_counter()
    edges = 0
    for source in sources:
        seen = {source}
        frontier = [source]
        for _ in range(hops):
            next_frontier: list[int] = []
            for vertex in frontier:
                for neighbour in adjacency.get(vertex, ()):
                    edges += 1
                    if neighbour not in seen:
                        seen.add(neighbour)
                        next_frontier.append(neighbour)
            frontier = next_frontier
    return time.perf_counter() - started, edges
