"""Resolving adapters and benchmark suites by name.

The core imports no concrete adapter: a system is chosen by name and built
through a factory, so adding one means adding a module and one registration
rather than editing the runner (TRD D2).

An adapter whose driver is not installed is not silently missing. It stays in
the listing with the reason it cannot be built, because "pgvector was not in
the list" and "psycopg is not installed" lead a user to different actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from theodb_bench.adapters.base import IndexSpec, SystemAdapter
from theodb_bench.adapters.fake import FakeAdapter
from theodb_bench.bench.vector import VectorWorkload
from theodb_bench.errors import AdapterError, ConfigError, ErrorContext, Phase

AdapterFactory = Callable[..., SystemAdapter]


@dataclass(frozen=True)
class AdapterEntry:
    """A registered system, and whether it can be built here."""

    name: str
    description: str
    factory: AdapterFactory
    requires: tuple[str, ...] = ()

    def unmet_requirements(self) -> list[str]:
        import importlib.util

        return [module for module in self.requires if importlib.util.find_spec(module) is None]

    @property
    def available(self) -> bool:
        return not self.unmet_requirements()

    def build(self, **kwargs: Any) -> SystemAdapter:
        missing = self.unmet_requirements()
        if missing:
            raise AdapterError(
                f"adapter {self.name!r} needs {', '.join(missing)}; "
                f"install it with: pip install 'theodb-bench[postgres]'",
                context=ErrorContext(phase=Phase.BOOTSTRAP, system=self.name),
            )
        adapter: SystemAdapter = self.factory(**kwargs)
        return adapter


def _postgres_factory(**kwargs: Any) -> SystemAdapter:
    from theodb_bench.adapters.postgres import PostgresAdapter

    return PostgresAdapter(**kwargs)


def _pgvector_factory(**kwargs: Any) -> SystemAdapter:
    from theodb_bench.adapters.postgres import PgvectorAdapter

    return PgvectorAdapter(**kwargs)


def _theodb_factory(**kwargs: Any) -> SystemAdapter:
    from theodb_bench.adapters.postgres import TheoDBAdapter

    return TheoDBAdapter(**kwargs)


ADAPTERS: Final[dict[str, AdapterEntry]] = {
    "fake": AdapterEntry(
        name="fake",
        description="Deterministic in-process system for testing the runner itself.",
        factory=FakeAdapter,
    ),
    "postgres": AdapterEntry(
        name="postgres",
        description="Upstream PostgreSQL, exact search only.",
        factory=_postgres_factory,
        requires=("psycopg",),
    ),
    "pgvector": AdapterEntry(
        name="pgvector",
        description="PostgreSQL with the pgvector extension (HNSW, IVFFlat).",
        factory=_pgvector_factory,
        requires=("psycopg",),
    ),
    "theodb": AdapterEntry(
        name="theodb",
        description="TheoDB: PostgreSQL 18 with the theodb_rs extension.",
        factory=_theodb_factory,
        requires=("psycopg",),
    ),
}


def get_adapter(name: str) -> AdapterEntry:
    if name not in ADAPTERS:
        raise ConfigError(
            f"unknown system {name!r}; known systems: {', '.join(sorted(ADAPTERS))}",
            context=ErrorContext(phase=Phase.PREFLIGHT),
        )
    return ADAPTERS[name]


@dataclass(frozen=True)
class BenchmarkEntry:
    """A registered benchmark suite."""

    id: str
    description: str
    workload: VectorWorkload
    default_repetitions: int = 3


BENCHMARKS: Final[dict[str, BenchmarkEntry]] = {
    "vector/synthetic/smoke": BenchmarkEntry(
        id="vector/synthetic/smoke",
        description=(
            "Small seeded synthetic corpus. Fast local validation of the whole "
            "pipeline; never a performance claim."
        ),
        workload=VectorWorkload(
            corpus_size=2_000,
            dimension=32,
            query_count=200,
            k=10,
            warmup_queries=20,
            indexes=(IndexSpec(kind="none"),),
        ),
        default_repetitions=1,
    ),
    "vector/synthetic/sweep": BenchmarkEntry(
        id="vector/synthetic/sweep",
        description=(
            "Seeded synthetic corpus with an HNSW ef_search sweep, producing a "
            "quality/throughput frontier rather than a single number."
        ),
        workload=VectorWorkload(
            corpus_size=10_000,
            dimension=64,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="none"), IndexSpec(kind="hnsw", parameters={"m": 16})),
            search_sweep={"ef_search": (16, 64, 256)},
        ),
        default_repetitions=3,
    ),
}


def get_benchmark(benchmark_id: str) -> BenchmarkEntry:
    if benchmark_id not in BENCHMARKS:
        raise ConfigError(
            f"unknown benchmark {benchmark_id!r}; known benchmarks: "
            f"{', '.join(sorted(BENCHMARKS))}",
            context=ErrorContext(phase=Phase.PREFLIGHT),
        )
    return BENCHMARKS[benchmark_id]
