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


def _alloydbomni_factory(**kwargs: Any) -> SystemAdapter:
    from theodb_bench.adapters.alloydb import AlloyDBOmniAdapter

    return AlloyDBOmniAdapter(**kwargs)


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
    "alloydbomni": AdapterEntry(
        name="alloydbomni",
        description=(
            "AlloyDB Omni: PostgreSQL with Google's scann access method. A query "
            "layer, not the managed service -- no disaggregated storage or read "
            "pool. The published image measured PostgreSQL 17."
        ),
        factory=_alloydbomni_factory,
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
    # The scann sweep is a separate suite for the same reason the hnsw one sweeps
    # `ef_search`: the search knob belongs to the index family, not to the
    # benchmark. `scann` takes `num_leaves_to_search`, `hnsw` takes `ef_search`,
    # and one suite cannot ask for both -- an adapter that cannot apply a
    # requested knob refuses the run rather than publishing one operating point
    # under several labels. The two families are compared at MATCHED RECALL from
    # their frontiers, never by pairing knob values that mean different things.
    # The AH variant exists because `quantizer` changes what is being measured,
    # not just how fast it is. Measured on the running server: AH is the
    # anisotropic quantizer ADR-0035 credits for the ~25x gap it found against the
    # ScaNN library, `CREATE INDEX ... WITH (quantizer='AH')` fails outright unless
    # `scann.enable_ah_quantizer` is on at build time, and the flag ships off. A
    # single scann suite would therefore have measured SQ8 and answered a question
    # about AH with it.
    "vector/sift/scann-ah": BenchmarkEntry(
        id="vector/sift/scann-ah",
        description=(
            "SIFT descriptors against AlloyDB's scann access method with its "
            "anisotropic AH quantizer, which is the configuration ADR-0035's gap "
            "was attributed to. Compared to an hnsw frontier at matched recall."
        ),
        workload=VectorWorkload(
            corpus_size=100_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="scann", parameters={"num_leaves": 316, "quantizer": "AH"}),),
            # The rerank depth is swept together with the leaves because the two
            # trade against each other, and pinning it at its -1 default would
            # measure a quantization-error ceiling instead of a frontier.
            search_sweep={
                "num_leaves_to_search": (5, 20, 80),
                "pre_reordering_num_neighbors": (100,),
            },
        ),
        default_repetitions=3,
    ),
    "vector/sift/hnsw": BenchmarkEntry(
        id="vector/sift/hnsw",
        description=(
            "SIFT descriptors against an hnsw access method at the same corpus "
            "size, queries and k as vector/sift/scann-ah, so the two frontiers "
            "can be read at matched recall."
        ),
        workload=VectorWorkload(
            corpus_size=100_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (16, 64, 256)},
        ),
        default_repetitions=3,
    ),
    # The 1M pair exists because scale is the decisive caveat on the 100k result:
    # ADR-0035's ~25x gap was measured at one million vectors, and an IVF-style
    # index and a graph index do not scale the same way. Comparing a 100k ratio to
    # a 1M ratio would compare two different experiments.
    "vector/sift1m/scann-ah": BenchmarkEntry(
        id="vector/sift1m/scann-ah",
        description=(
            "SIFT1M in full against AlloyDB's scann access method with the AH "
            "quantizer and exact-distance rescoring. The scale ADR-0035 measured."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="scann", parameters={"num_leaves": 1000, "quantizer": "AH"}),),
            search_sweep={
                "num_leaves_to_search": (20, 80),
                "pre_reordering_num_neighbors": (100,),
            },
        ),
        default_repetitions=3,
    ),
    "vector/sift1m/hnsw": BenchmarkEntry(
        id="vector/sift1m/hnsw",
        description=(
            "SIFT1M in full against an hnsw access method, at the same corpus "
            "size, queries and k as vector/sift1m/scann-ah."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64, 256)},
        ),
        default_repetitions=3,
    ),
    "vector/synthetic/scann-sweep": BenchmarkEntry(
        id="vector/synthetic/scann-sweep",
        description=(
            "Seeded synthetic corpus with a scann num_leaves_to_search sweep, "
            "producing the quality/throughput frontier of AlloyDB's own access "
            "method. Compared to an hnsw frontier at matched recall, never "
            "knob-to-knob."
        ),
        workload=VectorWorkload(
            corpus_size=10_000,
            dimension=64,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(
                IndexSpec(kind="none"),
                # num_leaves is passed, not derived: deriving it from cardinality
                # is a heuristic nobody here has measured.
                IndexSpec(kind="scann", parameters={"num_leaves": 100, "quantizer": "sq8"}),
            ),
            search_sweep={"num_leaves_to_search": (1, 10, 50)},
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
