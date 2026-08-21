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
from theodb_bench.bench.graph import GraphWorkload
from theodb_bench.bench.protocol import Workload
from theodb_bench.bench.retrieval import RetrievalWorkload
from theodb_bench.bench.vector import VectorWorkload
from theodb_bench.errors import AdapterError, ConfigError, ErrorContext, Phase
from theodb_bench.load import LoadModel

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
    # B-093 — o pilar lexical medido contra JULGAMENTO HUMANO, e não contra corpus semeado.
    #
    # O `bench/retrieval.py` traz nDCG@10, Recall@k, MRR e quatro pipelines desde sempre, e estava
    # na
    # lista de órfãos de `tests/test_module_reachability.py`: **nenhum benchmark registrado o
    # alcançava**. A consequência foi concreta — todo número lexical que este projeto publicou saiu
    # de
    # script ad-hoc, e o `m186` chegou a atribuir ao PRODUTO um limite (`bm25_search` só aceitaria
    # um
    # termo) que era do script, e que o [[B-014]] depois mediu ser falso.
    #
    # O corpus é o SciFact do BEIR: 5183 documentos, 300 consultas com qrel no split `test` — o
    # mesmo
    # que o `m186` usou, agora dentro do arnês e com o sha256 do arquivo publicado verificado.
    #
    # SÓ A PERNA LEXICAL, e isso é declarado e não uma limitação escondida: o BEIR publica texto e
    # julgamentos, não embeddings. Preencher os vetores com ruído faria as pernas densa e híbrida
    # RODAREM e os números PARECEREM medidos.
    "retrieval/scifact/lexical": BenchmarkEntry(
        id="retrieval/scifact/lexical",
        description=(
            "SciFact (BEIR) against the lexical pipeline, scored by nDCG@10 "
            "against human qrels rather than against a computed oracle. Dense "
            "and hybrid legs stay out: BEIR publishes no embeddings, and filling "
            "them with noise would make those numbers merely look measured."
        ),
        workload=RetrievalWorkload(
            corpus_size=5183,
            query_count=300,
            k=10,
            n=50,
            warmup_queries=20,
            pipelines=("lexical",),
        ),
        default_repetitions=3,
    ),
    # B-046 / B-042 — a fronteira larga, para comparar DOIS motores a RECALL CASADO.
    #
    # Comparar QPS a `ef_search` igual compara a coisa errada: `ef` não é a mesma unidade em dois
    # grafos diferentes. O [[B-046]] mede que, no mesmo `ef=64`, o TheoDB entrega recall 0,9600 e o
    # pgvector 0,9835 — logo um "déficit de QPS" lido nesse par está comparando um sistema que
    # buscou menos com um que buscou mais. A comparação honesta lê os dois na MESMA altura de
    # recall, e para isso é preciso ter pontos suficientes dos dois lados para interpolar.
    #
    # {40, 64, 128, 256} num único build: `ef_search` é GUC de sessão, então os quatro pontos saem
    # do mesmo índice e o build é pago uma vez. O 128 está aqui porque é onde o [[B-046]] mediu que
    # o TheoDB alcança o recall do pgvector em 64.
    #
    # E o build é medido junto, o que responde o [[B-042]] sem uma segunda corrida: o artefato já
    # registra `build_seconds` e `index_size_bytes` por ponto.
    "vector/sift1m/frontier": BenchmarkEntry(
        id="vector/sift1m/frontier",
        description=(
            "SIFT1M across a wide ef_search sweep, so two engines can be read at "
            "MATCHED RECALL instead of at matched ef -- which is not the same "
            "knob in two different graphs. Build time and index size come with it."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (40, 64, 128, 256)},
        ),
        default_repetitions=3,
    ),
    # B-018 — a pergunta é sobre o DEFAULT, e por isso a varredura é {40, 64} e não {64, 256}.
    #
    # Medido em 2026-08-21: o `theodb_hnsw` usa `ef_search = 64` por default (herdado do
    # `SCAN_EF` fixo pré-M35) e o pgvector usa 40. Em 64 o planner larga o índice numa junção
    # com filtro seletivo — e o pgvector, no MESMO 64, produz plano e custos idênticos aos
    # nossos. Não é defeito de implementação; é a escolha do default. Baixá-lo para 40 é uma
    # linha, e **troca recall por plano**.
    #
    # Este benchmark existe para que essa troca seja MEDIDA antes de decidida. Os dois valores
    # da varredura não são arbitrários: são exatamente os dois defaults em disputa. Um sweep
    # {64, 256} — o das outras suítes — responde outra pergunta (onde fica a fronteira) e não
    # esta (o que o default custa). `wiki/benchmarks/b018-planner-hnsw-juncao.md`.
    "vector/sift1m/ef-default": BenchmarkEntry(
        id="vector/sift1m/ef-default",
        description=(
            "SIFT1M against theodb_hnsw at the two ef_search DEFAULTS in dispute "
            "-- 40 (pgvector's) and 64 (ours). Measures what lowering the default "
            "costs in recall, which is the trade the B-018 fix pays."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (40, 64)},
        ),
        default_repetitions=3,
    ),
    # Query shapes beyond "top-10, one vector at a time", which is what eleven of
    # the twelve suites asked before these existed.
    #
    # k, because the graph descent and the rescore pool both scale with it and
    # not the same way: a system fast at k=10 can fall over at the k=100 a
    # reranking pipeline asks for.
    "vector/sift1m/k-sweep": BenchmarkEntry(
        id="vector/sift1m/k-sweep",
        description=(
            "SIFT1M against theodb_hnsw at k in {1, 10, 100}. The oracle is "
            "computed once at the largest k and sliced."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            k_sweep=(1, 10, 100),
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64, 256)},
        ),
        default_repetitions=3,
    ),
    # A filter, because it is the hardest case for a graph index -- the filter can
    # disconnect it -- and the easiest to answer fast and wrongly. The oracle
    # filters too, which is the only thing that tells "fast and right" apart from
    # "fast and crossing the filter".
    "vector/sift1m/filtered": BenchmarkEntry(
        id="vector/sift1m/filtered",
        description=(
            "SIFT1M partitioned into 100 tenants, every query filtered to one, "
            "and recall scored against a filtered oracle."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            filter_cardinality=100,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64, 256)},
        ),
        default_repetitions=3,
    ),
    # A batch, because one trip carrying many probes is what an agent's step
    # issues, and it is where per-query overhead stops dominating.
    "vector/sift1m/batch": BenchmarkEntry(
        id="vector/sift1m/batch",
        description=(
            "SIFT1M with 10 probes per round trip. Throughput is batches per "
            "second; a batch that costs as much as ten singles has no batching."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=500,
            k=10,
            batch_size=10,
            warmup_queries=50,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64,)},
        ),
        default_repetitions=3,
    ),
    # Throughput under contention, which is the regime a database actually runs
    # in and the one nothing here could measure until the load engine existed.
    #
    # Two suites, because they answer different questions and conflating them is
    # how throughput numbers become unfalsifiable:
    #
    #   `concurrency` sweeps the CLIENT COUNT in a closed loop. It answers "how
    #   far does this scale before it stops getting faster", and its latency is
    #   service time -- with no schedule there is nothing to be late for.
    #
    #   `saturation` fixes the clients and sweeps the ARRIVAL RATE. It answers
    #   "where does the queue start", which a closed loop cannot ask: a stalled
    #   system simply receives fewer requests and the stall never enters the
    #   distribution. Its response time includes the queueing, and the gap
    #   between the two latencies IS the queue.
    "vector/sift1m/concurrency": BenchmarkEntry(
        id="vector/sift1m/concurrency",
        description=(
            "SIFT1M against theodb_hnsw with a client population, closed loop. "
            "Sweeps the client count to find where throughput stops scaling."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=2_000,
            k=10,
            warmup_queries=100,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64,)},
            load=LoadModel(clients=16),
        ),
        default_repetitions=3,
    ),
    "vector/sift1m/saturation": BenchmarkEntry(
        id="vector/sift1m/saturation",
        description=(
            "SIFT1M against theodb_hnsw under Poisson arrivals at a fixed client "
            "count. Finds where the queue starts, which a closed loop cannot see."
        ),
        workload=VectorWorkload(
            corpus_size=1_000_000,
            dimension=128,
            query_count=4_000,
            k=10,
            warmup_queries=200,
            indexes=(IndexSpec(kind="hnsw", parameters={"m": 16}),),
            search_sweep={"ef_search": (64,)},
            load=LoadModel(clients=32, arrival_rate=2_000.0),
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


# B-058 bullet 2 — ABAIXO DE QUANTAS LINHAS o nosso colunar perde para o heap?
#
# O item registra o crossover do CONCORRENTE (o avaliador do AlloyDB mediu a inversao em algumas
# centenas de milhares: 31,6 ms colunar contra 27,5 ms heap a 100K) e diz, sobre o NOSSO: nunca
# medido, so estimado. Um colunar so paga o custo de decodificar stripe quando ha linha bastante
# para amortiza-lo; abaixo disso ele PERDE, e saber onde e o que decide quando liga-lo.
#
# A faixa vai de 10K a 2M porque contem a inversao esperada com folga dos DOIS lados. Comecar em
# 100K correria o risco de a curva nascer ja do lado errado do joelho, e a medicao concluir "sempre
# perde" ou "sempre ganha" por escolha de faixa — que e um resultado sobre a faixa, nao sobre o
# motor.
#
# Cada N RECARREGA os dados nos tres caminhos, entao esta corrida e cara por construcao. E o preco
# de uma CURVA; um ponto so nao responde "abaixo de quantas".
# B-007 — o pilar de grafo contra o que o usuario faria SEM nos.
#
# Medido pelo M184: 23 funcoes de grafo no binario default e 35 testes — a maior superficie publica
# depois do vetorial. E **nenhum artefato comparando o pilar com qualquer sistema**, nem um numero
# de latencia publicado. Qualquer afirmacao sobre o grafo hoje, em qualquer direcao, e sem lastro.
#
# O baseline e `WITH RECURSIVE` no proprio Postgres, e a escolha e do item: e o que o usuario faria
# sem a extensao. Ela responde a pergunta que ele tem — *vale a pena instalar isto?* — e nao a de
# quem ja decidiu usar um banco de grafo.
#
# Tres travessias (1, 2 e 3 saltos) e nao duas: o DoD pede "ao menos duas", e a terceira e barata e
# e onde a diferenca entre um CSR e uma juncao recursiva deveria aparecer, se aparecer. Uma
# comparacao que so olha 1 salto mediria quase so round-trip.
BENCHMARKS["graph/synthetic/vs-recursive-sql"] = BenchmarkEntry(
    id="graph/synthetic/vs-recursive-sql",
    description=(
        "Traversals over a seeded graph, each run twice: once through the CSR and "
        "once through plain `WITH RECURSIVE` on the same edge table, same server, "
        "same MVCC. The baseline is what a user would write without the extension."
    ),
    workload=GraphWorkload(
        vertex_count=200_000,
        average_degree=8,
        query_count=100,
        workloads=("1_hop", "2_hop", "3_hop"),
        compare_recursive_sql=True,
    ),
    default_repetitions=3,
)


# B-043 — ONDE a vazao lexical para de subir, e por que.
#
# Medido em 2026-08-13: o QPS satura em ~20 clientes numa maquina de 16 vCPU — de 20 a 80 o
# throughput nao cresce 1% e a p99 cresce 4x. A causa NAO esta medida, e uma das tres candidatas e o
# proprio cliente Python do arnes.
#
# Este benchmark existe para que a curva saia do arnes. O DoD do item exige, alem dela, um gerador
# EXTERNO (`pgbench`) na mesma maquina e corpus — se os dois saturarem no mesmo ponto, o
# cliente esta
# absolvido e o teto e do servidor; se o pgbench continuar subindo, o teto e nosso. Uma curva so, de
# um gerador so, nao separa as duas.
BENCHMARKS["retrieval/scifact/concurrency"] = BenchmarkEntry(
    id="retrieval/scifact/concurrency",
    description=(
        "SciFact lexical under a client population, closed loop, sweeping the client "
        "count to find where throughput stops scaling. Quality is not reported under "
        "concurrency: the queries wrap around, so an average nDCG would be over a "
        "repeated set rather than over the judged one."
    ),
    workload=RetrievalWorkload(
        corpus_size=5183,
        query_count=300,
        k=10,
        n=50,
        warmup_queries=20,
        pipelines=("lexical",),
        client_sweep=(1, 5, 10, 20, 40, 80),
    ),
    default_repetitions=3,
)


BENCHMARKS["analytical/crossover/row-count"] = BenchmarkEntry(
    id="analytical/crossover/row-count",
    description=(
        "The same aggregations over heap, columnar and Parquet across a row-count "
        "sweep, to find where columnar stops losing to heap. A single row count "
        "cannot answer 'below how many rows', which is the question."
    ),
    workload=AnalyticalWorkload(
        row_count=10_000,
        row_count_sweep=(10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000),
    ),
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
