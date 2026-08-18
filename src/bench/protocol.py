"""What the orchestrator needs from a workload, and nothing more.

The eleven-phase runner used to name `VectorWorkload` in its request type and
construct `VectorBenchmark` by hand, so a second workload family could not be run
at all. Measured 2026-08-17, three of them were affected: `bench/analytical.py`,
`bench/graph.py` and `bench/retrieval.py` were implemented and had zero importers
in `src/` — the analytical one complete with its own oracle, adapter methods and a
four-state residency gate, and no way to invoke it.

Four questions carried the coupling, and each is one a workload family should
answer about itself: which benchmark to build, what the benchmark artefact says,
how many operations a run should have produced, and how many of them were warm-up.
They are protocol members here, so the orchestrator depends on the abstraction and
a new family is a module rather than an edit to the runner
(`rules/architecture.md § 2`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Benchmark(Protocol):
    """A measurable workload, already bound to its data."""

    def load(self, adapter: Any) -> float | None:
        """Put the data in place. Seconds taken, or None when nothing was loaded."""
        ...

    def points(
        self, adapter: Any, repetitions: int, make_client: Callable[[], Any] | None = None
    ) -> list[Any]:
        """Every configuration measured, as PointResults.

        `make_client` opens one connection per client when the workload declares
        a client population. A benchmark that only ever issues work serially may
        ignore it, but it must accept it: the runner has no way to know which
        kind it holds, and asking would put the regime back in the caller.
        """
        ...


@runtime_checkable
class Workload(Protocol):
    """A declarative workload: what to measure, before any data exists."""

    def build(self, corpus: Any, queries: Any) -> Benchmark:
        """The benchmark for this workload, bound to its data.

        `corpus` and `queries` are the verified dataset when one was supplied and
        None otherwise; a family that generates its own data from a seed ignores
        them.
        """
        ...

    def benchmark_payload(self) -> dict[str, Any]:
        """The artefact fields this workload owns: `workload`, `quality`, `parameters`.

        Written by the family rather than by the runner. `k` and `operation_count`
        mean different things to a k-NN probe and to an analytical query, and the
        quality axis differs outright: approximate retrieval is scored by recall,
        an analytical answer is right or wrong. A runner filling these in would
        describe one family in the other's vocabulary.
        """
        ...

    def expected_operations(self, measured_points: int, repetitions: int) -> int:
        """How many operations a complete run should have produced.

        The validation gate compares this to what was observed, so the unit has to
        come from whoever knows it: a query per probe here, a query per execution
        path there.
        """
        ...

    @property
    def warmup_operations(self) -> int:
        """Operations run untimed before measurement began."""
        ...

    def quality_was_reported(self, points: list[Any]) -> bool:
        """Whether this run carries the quality axis this family is scored on.

        Asked of the family because the axis differs in kind, not just in name:
        approximate retrieval reports a recall per repetition, and an analytical
        answer is right or wrong once. A runner checking `recall is not None`
        would mark every analytical run as quality-less and invalidate it for
        failing to report a number it has no business producing.
        """
        ...
