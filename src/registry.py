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
from theodb_bench.bench.analytical import AnalyticalWorkload
from theodb_bench.bench.protocol import Workload
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
    """A registered benchmark suite.

    `workload` is typed by the protocol rather than by one family: the
    orchestrator asks a workload to build its own benchmark, so registering a
    second family is a module and an entry rather than a change to the runner.
    """

    id: str
    description: str
    workload: Workload
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
    # The reference scale. 20 000 000 real SIFT descriptors is one order of
    # magnitude past everything else registered here, and it is chosen from
    # measured size rather than ambition: 1.27 GB per million (heap plus HNSW,
    # measured on the host) puts 20M at 25.4 GB, which is 9% of that disk. 100M
    # fits too and turns the build into hours of work; ~200M is the physical
    # ceiling and leaves no room for maintenance_work_mem or sort spill.
    #
    # The corpus streams. It is 2.64 GB on disk as uint8 and 10.2 GB as the
    # float32 the oracle computes over, on a host that also runs the database
    # under test, so it arrives as a `CorpusSource` and the oracle carries a
    # running top-k across chunks (`bench/corpus.py`, `streaming.py`).
    #
    # Ground truth is computed, not read. BIGANN publishes neighbour ids for the
    # full billion; against a 20M prefix they name rows that do not exist.
    #
    # Two searches, not a sweep: at this scale each point is a full pass over
    # 10 000 queries, and a sweep would be measured before it was known that one
    # point completes. Widen it once there is a build time on record.
    "vector/bigann20m/hnsw": BenchmarkEntry(
        id="vector/bigann20m/hnsw",
        description=(
            "20 000 000 real SIFT descriptors from BIGANN against theodb_hnsw. The "
            "reference scale: one order of magnitude past SIFT1M, with the corpus "
            "streamed and ground truth computed over it."
        ),
        workload=VectorWorkload(
            corpus_size=20_000_000,
            dimension=128,
            query_count=1_000,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64, 256)},
        ),
        default_repetitions=3,
    ),
    # The load-only companion. `--index none` is not a knob, so measuring the
    # streamed load of 20M without paying for a graph build needs its own entry:
    # the load is what the streaming work changed, and it is measurable in minutes
    # where the build is measurable in hours.
    "vector/bigann20m/load": BenchmarkEntry(
        id="vector/bigann20m/load",
        description=(
            "The streamed load of 20 000 000 real SIFT descriptors, with no index "
            "built. Isolates the load path from the build."
        ),
        workload=VectorWorkload(
            corpus_size=20_000_000,
            dimension=128,
            query_count=100,
            k=10,
            query_cap=100,
            indexes=(IndexSpec(kind="none"),),
        ),
        default_repetitions=1,
    ),
    # The pair that answers B-057, and it exists because the first attempt at that
    # answer compared the wrong index on our side. TheoDB's ScaNN-class path is
    # `theodb_ivfflat` with the anisotropic quantizer (`pq_subspaces`), the LUT16
    # width (`pq_bits=4`) and the exact-distance second stage
    # (`separate_storage=1, refine=1`) -- the arc is called pg_scann internally.
    # Racing `theodb_hnsw` against AlloyDB's scann compares our graph index to
    # their quantized IVF index, which is a real comparison and not the one the
    # item asks for.
    #
    # Both stages are matched deliberately. Probe depth: `probes` against
    # `num_leaves_to_search`. Rescore pool: ours is `64 * over_fetch`, so
    # `over_fetch=2` gives 128 candidates against their 100 -- the closest the two
    # knobs reach, and the artefact records both so a reader can see they are not
    # identical.
    "vector/sift/pg-scann": BenchmarkEntry(
        id="vector/sift/pg-scann",
        description=(
            "SIFT descriptors against TheoDB's own ScaNN-class path: theodb_ivfflat "
            "with the anisotropic quantizer, LUT16 codes and the exact-distance "
            "second stage. The counterpart to vector/sift/scann-ah."
        ),
        workload=VectorWorkload(
            corpus_size=100_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(
                IndexSpec(
                    kind="ivfflat",
                    parameters={
                        # lists ~ sqrt(rows), the same rule the competitor's
                        # num_leaves=316 follows, so the partitioning is comparable.
                        "lists": 316,
                        # 64 subspaces over 128 dimensions is 2 dimensions per
                        # subspace, the width ScaNN and FAISS use. Measured on
                        # 2026-08-17 at probes=20: 16 subspaces caps recall at
                        # 0.8172, 32 gives 0.9270 and 64 gives 0.9570, at
                        # indistinguishable throughput. The first frontier taken
                        # here used 16 and reported a ceiling that belonged to the
                        # quantizer rather than to the index.
                        "pq_subspaces": 64,
                        "pq_bits": 4,
                        "separate_storage": 1,
                        "refine": 1,
                    },
                ),
            ),
            search_sweep={"probes": (5, 20, 80), "over_fetch": (2,)},
        ),
        default_repetitions=3,
    ),
    # Quantizer width is a build parameter, so it is swept as separate index
    # configurations rather than as a search knob. It exists because the first
    # pg-scann frontier plateaued at 0.82 recall while its rescore pool was
    # verified at 128 candidates for k=10 -- 128 exact rescores cannot cap recall
    # at 0.82 unless the true neighbours are absent from the candidate set, so the
    # ceiling is stage-1 quantization error. `pq_subspaces=16` over 128 dimensions
    # is 8 dimensions per subspace; ScaNN and FAISS use 2. The sweep measures that
    # instead of assuming it.
    "vector/sift/pg-scann-quantizer": BenchmarkEntry(
        id="vector/sift/pg-scann-quantizer",
        description=(
            "SIFT descriptors against TheoDB's pg_scann path at three quantizer "
            "widths, to locate the operating point before any cross-engine "
            "comparison is made. Coarse codes cap recall no rescore pool can lift."
        ),
        workload=VectorWorkload(
            corpus_size=100_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=tuple(
                IndexSpec(
                    kind="ivfflat",
                    parameters={
                        "lists": 316,
                        "pq_subspaces": subspaces,
                        "pq_bits": 4,
                        "separate_storage": 1,
                        "refine": 1,
                    },
                )
                # 8, 4 and 2 dimensions per subspace over 128 dims.
                for subspaces in (16, 32, 64)
            ),
            search_sweep={"probes": (20,), "over_fetch": (2,)},
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


BENCHMARKS["analytical/synthetic/paths"] = BenchmarkEntry(
    id="analytical/synthetic/paths",
    description=(
        "The same seeded rows stored three ways -- heap, columnar, Parquet -- with "
        "four aggregations run against each. The answer is checked against this "
        "benchmark's own oracle, so a path that is fast and wrong is caught."
    ),
    workload=AnalyticalWorkload(row_count=200_000),
    default_repetitions=3,
)


def get_benchmark(benchmark_id: str) -> BenchmarkEntry:
    if benchmark_id not in BENCHMARKS:
        raise ConfigError(
            f"unknown benchmark {benchmark_id!r}; known benchmarks: "
            f"{', '.join(sorted(BENCHMARKS))}",
            context=ErrorContext(phase=Phase.PREFLIGHT),
        )
    return BENCHMARKS[benchmark_id]
