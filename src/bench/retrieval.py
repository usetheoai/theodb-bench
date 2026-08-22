"""The retrieval workload: lexical, dense, hybrid RRF, and hybrid plus rerank.

All four pipelines run over **the same corpus and the same query set**. That is
not a convenience — comparing pipelines evaluated on different data measures
the data.

Two rules this module exists to enforce:

Quality and performance are separate axes. A pipeline is described by nDCG@10,
Recall@k and MRR *together with* throughput and latency. Throughput alone
cannot distinguish a fast pipeline from one returning worse answers faster.

Model time is never charged to the database. A reranking pipeline reports the
model's stage separately, so the database's contribution stays visible.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from theodb_bench.adapters.base import (
    Document,
    DocumentTableSpec,
    HybridQuery,
    KnnQuery,
    LexicalQuery,
    SystemAdapter,
)
from theodb_bench.analysis.fusion import fuse_to_ids
from theodb_bench.analysis.quality import mrr_at_k, ndcg_at_k, recall_at_n
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.errors import (
    BenchError,
    ConfigError,
    ErrorContext,
    MeasurementError,
    Phase,
    SystemUnavailableError,
    UnsupportedCapabilityError,
)
from theodb_bench.load import LoadModel, client_pool, run_load

DEFAULT_TABLE: Final[str] = "bench_documents"

LEXICAL: Final[str] = "lexical"
VECTOR: Final[str] = "vector"
HYBRID_RRF: Final[str] = "hybrid_rrf"
HYBRID_RRF_RERANK: Final[str] = "hybrid_rrf_rerank"

PIPELINES: Final[tuple[str, ...]] = (LEXICAL, VECTOR, HYBRID_RRF, HYBRID_RRF_RERANK)

#: Qual capacidade cada perna exercita. Publico porque a matriz de capacidades deriva dele:
#: contar uma suite como medindo `hybrid` pelo PREFIXO do id acertaria por acidente e erraria
#: no dia em que uma suite `retrieval/*` declarasse so a perna lexical — que e o caso do
#: `scifact`. A declaracao nao mente; o nome e convencao.
PIPELINE_CAPABILITY: Final[dict[str, str]] = {
    LEXICAL: "lexical",
    VECTOR: "vector_exact",
    HYBRID_RRF: "hybrid",
    HYBRID_RRF_RERANK: "rerank",
}

_VOCABULARY: Final[tuple[str, ...]] = (
    "vector",
    "index",
    "recall",
    "latency",
    "throughput",
    "storage",
    "query",
    "planner",
    "cache",
    "buffer",
    "replication",
    "durability",
    "checkpoint",
    "segment",
    "shard",
    "partition",
    "embedding",
    "token",
    "corpus",
    "ranking",
    "fusion",
    "rerank",
    "traversal",
    "columnar",
    "parquet",
    "graph",
    "agent",
    "memory",
)


@dataclass(frozen=True)
class RetrievalWorkload:
    """A declarative retrieval workload."""

    corpus_size: int
    query_count: int
    dimension: int = 64
    k: int = 10
    n: int = 50
    """Candidate depth each leg retrieves before fusion."""

    metric: str = "cosine"
    seed: int = 20260813
    table: str = DEFAULT_TABLE
    pipelines: tuple[str, ...] = PIPELINES
    rrf_k: int = 60
    warmup_queries: int = 0

    #: Contagens de clientes a varrer, em laco fechado. Vazio = um cliente, que e o que as suites
    #: existentes usam.
    #:
    #: Existe para o [[B-043]]: o QPS lexical satura em ~20 clientes numa maquina de 16 vCPU e nao
    #: sobe mais — de 20 a 80 o throughput nao cresce 1% e a p99 cresce 4x. **A causa nao esta
    #: medida**, e uma das tres candidatas e o proprio cliente Python do arnes. Sem o arnes saber
    #: dirigir N clientes, essa hipotese nao pode nem ser posta ao lado de um gerador externo.
    client_sweep: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.pipelines) - set(PIPELINES)
        if unknown:
            raise ConfigError(
                f"unknown pipeline(s): {', '.join(sorted(unknown))}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )
        if self.k > self.n:
            raise ConfigError(
                f"k={self.k} exceeds the candidate depth n={self.n}",
                context=ErrorContext(phase=Phase.PREFLIGHT),
            )

    def table_spec(self) -> DocumentTableSpec:
        return DocumentTableSpec(table=self.table, dimension=self.dimension, metric=self.metric)

    # ---- protocolo `Workload` (bench/protocol.py) ----
    #
    # Estes cinco membros faltavam, e a consequência foi medida: `bench.retrieval` estava na lista
    # de
    # órfãos de `tests/test_module_reachability.py` — a pipeline inteira existia e NENHUM benchmark
    # registrado a alcançava. Todo número lexical publicado saiu de script ad-hoc, e o `m186` chegou
    # a
    # atribuir ao produto um limite que era do script.

    def build(self, corpus: Any, queries: Any) -> RetrievalBenchmark:
        """O benchmark ligado aos seus dados — reais quando houver, semeados quando não.

        `corpus`/`queries` vêm `None` quando ninguém passou `--dataset`, e aí o corpus semeado do
        `generate_corpus` entra. Ele exercita a pipeline e as métricas; **não é alegação de
        qualidade**, e um artefato produzido sobre ele não deve ser citado como tal.
        """
        if corpus is None or queries is None:
            return RetrievalBenchmark(self)
        return RetrievalBenchmark(self, documents=corpus, queries=queries)

    def benchmark_payload(self) -> dict[str, Any]:
        return {
            "workload": {
                "type": "retrieval",
                "loop": "closed",
                "clients": max(self.client_sweep) if self.client_sweep else 1,
                "arrival_rate_per_second": None,
                "k": [self.k],
                "operation_count": self.query_count,
            },
            # A qualidade aqui é JULGADA, não computada: o nDCG mede contra qrels humanos, e não
            # contra um oráculo que o próprio arnês calcula. Chamar os dois de "ground_truth:
            # computed" apagaria a diferença que mais importa entre medir recall aproximado e medir
            # qualidade de busca.
            # `dataset` e nao `computed`: os qrels VEM COM o corpus, humanos, em vez de serem
            # calculados pelo arnes como o oraculo de recall e. O schema do artefato ja fazia
            # essa distincao — a primeira versao disto inventou `judged`, o schema recusou, e
            # estava certo: a palavra que faltava ja existia, e com melhor nome.
            "quality": {"metric": "ndcg", "ground_truth": "dataset"},
            "parameters": {
                "pipelines": list(self.pipelines),
                "n": [self.n],
                "rrf_k": [self.rrf_k],
                **({"clients": list(self.client_sweep)} if self.client_sweep else {}),
            },
        }

    def expected_operations(self, measured_points: int, repetitions: int) -> int:
        """Quantas operacoes uma corrida completa deveria ter produzido.

        Com varredura de clientes, cada ponto emite `query_count` operacoes POR CLIENTE — ver
        `RetrievalBenchmark.run_concurrent` para por que a contagem escala. Somar `query_count` por
        ponto contaria o ponto de 80 clientes como se ele tivesse feito o trabalho do de 1, e o
        portao `operation_count` reprovaria a corrida inteira por uma conta errada dele proprio.
        """
        if not self.client_sweep:
            return measured_points * repetitions * self.query_count
        por_pipeline = sum(self.client_sweep) * self.query_count
        return len(self.pipelines) * repetitions * por_pipeline

    @property
    def warmup_operations(self) -> int:
        return self.warmup_queries

    def quality_was_reported(self, points: list[Any]) -> bool:
        return any(
            repetition.ndcg_at_10 is not None
            for point in points
            for repetition in point.repetitions
        )


@dataclass(frozen=True)
class QuerySet:
    """Queries with their relevance judgements."""

    texts: tuple[str, ...]
    vectors: npt.NDArray[np.float32]
    relevance: tuple[dict[int, float], ...]
    """Graded relevance per query, document id to gain."""

    def relevant_ids(self, index: int) -> set[int]:
        return {doc_id for doc_id, gain in self.relevance[index].items() if gain > 0}


def _vocabulario_para(corpus_size: int) -> tuple[str, ...]:
    """O vocabulário cresce com o corpus, para a perna lexical não perder o que ordenar.

    O gerador declara que as duas pernas **concordam no documento primário** — é a situação
    que a fusão existe para explorar. Com vocabulário FIXO essa intenção vale a corpus
    pequeno e quebra com a escala. Medido em 2026-08-22, documentos que contêm a consulta
    inteira, em média, com os 28 termos originais:

        corpus=  200 ->  1,6        corpus= 1000 ->  3,9
        corpus=  500 ->  3,3        corpus= 5000 -> 38,9

    A 5.000 o BM25 enfrentava ~39 candidatos igualmente casados para 1 relevante, e a perna
    lexical marcou **nDCG@10 = 0,0752** contra 0,8266 da vetorial. **Não era defeito do motor
    lexical: era o corpus não ter sinal.** E a fusão, fundindo uma perna forte com ruído,
    piorou a perna forte — comportamento correto do RRF sobre uma pergunta que ninguém
    deveria fazer.

    A raiz cúbica vem da forma do problema: um documento tem ~8 termos, a consulta tem ~3, e
    a chance de conter os três cai com o cubo da razão termos-por-documento sobre vocabulário.
    Para o número esperado de casamentos ficar constante enquanto `N` cresce, o vocabulário
    tem de crescer com `N^(1/3)`. Os 28 termos originais continuam sendo o piso e o começo da
    lista — corpus pequeno gera exatamente o corpus que gerava antes.
    """
    alvo = max(len(_VOCABULARY), int(8 * (max(corpus_size, 1) / 2) ** (1 / 3)))
    if alvo <= len(_VOCABULARY):
        return _VOCABULARY
    # Termos gerados além do vocabulário natural. O corpus já se declara sintético e não
    # parecido com linguagem natural; o que ele precisa é de poder de discriminação.
    extras = tuple(f"termo{i:04d}" for i in range(alvo - len(_VOCABULARY)))
    return _VOCABULARY + extras


def veredito_de_qualidade(
    nome_a: str,
    ndcg_a: Mapping[int, float],
    nome_b: str,
    ndcg_b: Mapping[int, float],
) -> str | None:
    """Teste pareado de qualidade entre dois caminhos, com a CONSULTA como unidade.

    É o que o [[B-005]] pede — *"uma fusão cujo ganho sobre o vetorial puro sobreviva a
    teste pareado de significância"* — e não era possível enquanto o nDCG por consulta era
    descartado. A repetição não serve de unidade: o nDCG é idêntico nas cinco repetições
    porque corpus e consultas são determinísticos.

    Reusa `compare.pair_by_query` e `render_paired_verdict`, que já existiam e já eram
    usados para latência. A única coisa que faltava era o dado.

    Devolve `None` quando não há consulta em comum — ausência de par é ausência de
    resposta, e um zero ali leria como empate medido.
    """
    from theodb_bench.compare import render_paired_verdict

    comuns = set(ndcg_a) & set(ndcg_b)
    if not comuns:
        return None
    # `lower_is_better=False` NAO é detalhe. `render_paired_verdict` nasceu para latência,
    # onde menor vence, e tem esse default. Medido em 2026-08-22, sem passá-lo: o veredito
    # imprimiu *"hybrid_rrf beats vector"* sobre `mean diff = -0,007`, com a fusão em 0,8195
    # e a vetorial em 0,8266 — **o oposto da medição**, e a frase parecia certa.
    return render_paired_verdict(
        nome_a,
        dict(ndcg_a),
        nome_b,
        dict(ndcg_b),
        metric="ndcg_at_10",
        lower_is_better=False,
    )


def generate_corpus(workload: RetrievalWorkload) -> tuple[list[Document], QuerySet]:
    """A seeded corpus, query set and judgement set.

    Judgements are constructed rather than assumed: each query is built from the
    terms of a small set of documents, and exactly those documents are graded
    relevant. That makes the ground truth exact for this corpus, which is what
    lets a pipeline defect show up as a quality drop rather than as noise.

    This is a synthetic corpus. It exercises the pipeline and the metrics; it
    does not resemble natural language, and no quality claim about a real
    system should be read from it.
    """
    rng = np.random.default_rng(workload.seed)
    vocabulario = _vocabulario_para(workload.corpus_size)
    documents: list[Document] = []
    doc_terms: list[set[str]] = []

    for doc_id in range(workload.corpus_size):
        term_count = int(rng.integers(6, 12))
        terms = list(rng.choice(vocabulario, size=term_count, replace=True))
        doc_terms.append(set(terms))
        vector = rng.standard_normal(workload.dimension).astype(np.float32)
        documents.append(Document(id=doc_id, text=" ".join(terms), vector=vector))

    corpus_vectors = np.vstack([document.vector for document in documents])

    texts: list[str] = []
    vectors: list[npt.NDArray[np.float32]] = []
    relevance: list[dict[int, float]] = []

    for _ in range(workload.query_count):
        # Two documents seed each query: one strongly relevant, one partially.
        primary, secondary = (
            int(i) for i in rng.choice(workload.corpus_size, size=2, replace=False)
        )
        primary_terms = sorted(doc_terms[primary])
        secondary_terms = sorted(doc_terms[secondary])
        chosen = primary_terms[: max(2, len(primary_terms) // 2)] + secondary_terms[:1]
        texts.append(" ".join(chosen))
        # The query vector sits near the primary document, with noise, so the
        # dense leg and the lexical leg agree on the primary and disagree
        # elsewhere -- which is the situation fusion exists for.
        noise = rng.standard_normal(workload.dimension).astype(np.float32) * 0.35
        vectors.append((corpus_vectors[primary] + noise).astype(np.float32))
        relevance.append({primary: 3.0, secondary: 1.0})

    return documents, QuerySet(
        texts=tuple(texts),
        vectors=np.vstack(vectors).astype(np.float32),
        relevance=tuple(relevance),
    )


@dataclass
class PipelineResult:
    """One pipeline over the whole query set, for one repetition."""

    pipeline: str
    repetition: int
    successes: int = 0
    errors: int = 0
    timeouts: int = 0
    duration_seconds: float = 0.0
    latency: LatencySummary | None = None
    ndcg_at_10: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    status: str = "measured"
    status_detail: str | None = None

    #: Custo do indice, quando a familia o conhece. O runner le os dois de toda repeticao, e ate
    #: aqui `PipelineResult` tinha oito dos dez campos que ele pede — a familia estava orfa, entao
    #: ninguem nunca tinha pedido os outros dois. Ficam `None` no lexical: o indice BM25 e
    #: construido
    #: pelo `bm25_build` dentro do adapter, e o arnes ainda nao le esse custo. **`None` diz "nao
    #: medido"; um zero diria "de graca", que e falso.**
    build_seconds: float | None = None
    index_size_bytes: int | None = None

    #: Latencia por consulta, em ms, indexada pela POSICAO da consulta no conjunto. O runner grava
    #: isto no artefato e o teste pareado do `significance.py` o consome — sem ele, comparar duas
    #: corridas so pode usar agregado, e agregado nao tem par. Este campo faltava, e a familia
    #: inteira estava orfa, entao nada nunca o pediu.
    latency_by_query: dict[int, float] = field(default_factory=dict)
    #: nDCG@10 por consulta, na mesma chave que `latency_by_query`.
    #:
    #: O nome é `quality_by_query` e não `ndcg_by_query` porque o runner o emite sem saber de
    #: qual família veio — a família vetorial reporta recall, esta reporta nDCG, e o artefato
    #: guarda "a qualidade que esta corrida mediu" com a métrica nomeada ao lado.
    #:
    #: Guardado e não só mediado porque o teste pareado de QUALIDADE precisa da consulta
    #: como unidade. Medido em 2026-08-22: o nDCG é IDÊNTICO nas cinco repetições — 0,8266
    #: cinco vezes — porque corpus e consultas são determinísticos. Qualidade não varia
    #: entre repetições; só a vazão varia. Um teste pareado entre repetições teria variância
    #: zero por construção, e o [[B-005]] pede justamente um teste pareado.
    #:
    #: Com isto, `compare.pair_by_query` e `render_paired_verdict` — que já existiam e já
    #: eram usados para latência — passam a servir também para qualidade.
    quality_by_query: dict[int, float] = field(default_factory=dict)

    @property
    def throughput(self) -> float | None:
        return self.successes / self.duration_seconds if self.duration_seconds > 0 else None

    @property
    def recall(self) -> float | None:
        """Alias de `recall_at_k` sob o nome que o relatorio pede.

        O `report.pareto_payload` le `r.recall` de toda repeticao — acoplamento ao vocabulario da
        familia vetorial, apesar de o protocolo `Workload` dizer que cada familia possui o
        seu. A
        ponte e honesta e nao um remendo: `recall_at_k` E recall, e a fronteira recall x vazao e
        leitura legitima tambem para retrieval. O que NAO se alia e o nDCG, que nao e recall e
        entraria como um numero de outra natureza no mesmo eixo.
        """
        return self.recall_at_k

    def metric_series(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        throughput = self.throughput
        if throughput is not None:
            series["throughput_per_second"] = [throughput]
        for name, value in (
            ("ndcg_at_10", self.ndcg_at_10),
            ("recall_at_k", self.recall_at_k),
            ("mrr", self.mrr),
        ):
            if value is not None:
                series[name] = [value]
        if self.latency is not None:
            for name in ("p50", "p95", "p99"):
                measured = getattr(self.latency, name)
                if isinstance(measured, float):
                    series[f"latency_{name}_ms"] = [measured]
        for stage, seconds in self.stage_seconds.items():
            series[f"stage_{stage}_seconds"] = [seconds]
        return series


@dataclass
class RetrievalPoint:
    """Uma configuração medida — aqui, um pipeline — com suas repetições.

    Mesma forma que o `PointResult` da família vetorial, porque é o que o runner lê: `label`,
    `parameters`, `status`, `repetitions` e `metric_series()`. Não é herança nem base compartilhada
    de propósito: as duas famílias reportam eixos de qualidade DIFERENTES (recall computado contra
    oráculo lá, nDCG contra julgamento humano aqui), e uma base comum convidaria a somá-los.
    """

    label: str
    parameters: dict[str, Any]
    status: str = "measured"
    status_detail: str | None = None
    repetitions: list[PipelineResult] = field(default_factory=list)

    def metric_series(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        for repeticao in self.repetitions:
            for nome, valores in repeticao.metric_series().items():
                series.setdefault(nome, []).extend(valores)
        return series


class RetrievalBenchmark:
    """Runs every declared pipeline over one corpus and one query set."""

    def __init__(
        self,
        workload: RetrievalWorkload,
        *,
        documents: Sequence[Document] | None = None,
        queries: QuerySet | None = None,
    ) -> None:
        self.workload = workload
        if documents is None or queries is None:
            self.documents, self.queries = generate_corpus(workload)
        else:
            # Corpus real. O `corpus_size`/`query_count` declarados no workload descrevem o SEMEADO;
            # com dado real quem manda é o dado, e forçar o declarado truncaria o corpus em
            # silêncio.
            self.documents, self.queries = list(documents), queries

    def load(self, adapter: SystemAdapter) -> float:
        outcome = adapter.load_documents(self.workload.table_spec(), self.documents)
        if not outcome.complete:
            raise MeasurementError(
                f"loaded {outcome.rows_loaded} of {outcome.rows_expected} documents",
                context=ErrorContext(phase=Phase.DATASET_LOAD, system=adapter.system_id),
            )
        return outcome.seconds

    def warm_up(self, adapter: SystemAdapter, pipeline: str) -> None:
        """Untimed queries, discarded."""
        for index in range(min(self.workload.warmup_queries, len(self.queries.texts))):
            try:
                self._run_one(adapter, pipeline, index)
            except BenchError:
                # A warm-up failure is not a measurement; the measured window
                # will surface the same problem where it can be counted.
                return

    def run_concurrent(
        self,
        adapter: SystemAdapter,
        pipeline: str,
        repetition: int,
        clients: int,
        make_client: Callable[[], SystemAdapter] | None,
    ) -> PipelineResult:
        """Uma passada da pipeline com POPULACAO de clientes, em laco fechado.

        B-043. O que ela mede que a serial nao mede: onde a vazao para de subir. Uma curva de
        cliente contra QPS e o unico jeito de distinguir "o trabalho por consulta e caro" de "ha
        fila contra capacidade fixa" — e a segunda so aparece com mais de um cliente.

        Reusa o `run_load`, que ja registra os dois relogios por requisicao e conta a falha em vez
        de descarta-la; um erro que encolhe a amostra em silencio transforma sistema quebrado em
        sistema rapido.
        """
        result = PipelineResult(pipeline=pipeline, repetition=repetition)
        capability = PIPELINE_CAPABILITY[pipeline]
        if not adapter.supports(capability):
            result.status = "unsupported"
            result.status_detail = f"{adapter.system_id} does not support {capability}"
            return result

        modelo = LoadModel(clients=clients)
        abrir, fechar = client_pool(adapter, make_client, clients)
        total = len(self.queries.texts)

        # OPERACOES POR CLIENTE, e nao um total fixo. MEDIDO em 2026-08-21, e a primeira versao
        # desta funcao errou exatamente aqui:
        #
        #   clientes | total FIXO em 300 | 300 POR cliente
        #          5 |             598,6 |           646,1
        #         20 |             570,2 |           801,8
        #         80 |         **277,7** |       **827,0**
        #
        # Com total fixo, 80 clientes dividem 300 operacoes — **3,75 cada** — e a abertura de
        # conexao domina a janela medida. A curva COLAPSA, e o colapso e do desenho da medicao, nao
        # do sistema. Com a contagem escalando, ela sobe e satura, que e o que se quer observar.
        #
        # Isso me fez publicar uma conclusao errada no [[B-043]] antes de medir de novo. Fica aqui
        # porque a proxima pessoa a mexer nisto vai sentir a mesma tentacao de fixar o total para
        # "comparar o mesmo trabalho" — e comparar o mesmo trabalho entre populacoes diferentes de
        # cliente e justamente o que nao se pode fazer em laco fechado.
        por_cliente = total
        operacoes = por_cliente * clients

        def emitir(cliente: SystemAdapter, indice: int) -> None:
            # `indice % total` porque o conjunto de consultas tem tamanho proprio: dar a volta mede
            # a mesma carga, e truncar mediria menos consulta com mais cliente.
            self._run_one(cliente, pipeline, indice % total)

        carga = run_load(
            abrir,
            emitir,
            count=operacoes,
            model=modelo,
            fatal=(SystemUnavailableError,),
            close_client=fechar,
        )
        result.successes = carga.successes
        result.errors = carga.errors
        result.duration_seconds = carga.duration_seconds

        # DOIS RELOGIOS, e a distincao e o achado que o B-043 persegue.
        #
        # `response_seconds` e o que o cliente ve — inclui a espera na fila. `service_seconds` e o
        # que o servidor levou. Se a resposta cresce e o servico fica PLANO, o teto e fila contra
        # capacidade fixa; se o servico tambem cresce, o servidor esta ficando mais lento. Reportar
        # so um dos dois deixaria o leitor concluir o que quiser, que e o que o item recusa.
        respostas = [r.response_seconds * 1000.0 for r in carga.requests if r.ok]
        servicos = [r.service_seconds * 1000.0 for r in carga.requests if r.ok]
        result.latency = summarise_latency(respostas)
        for i, valor in enumerate(respostas):
            result.latency_by_query[i] = valor
        if servicos:
            # `stage_seconds` e o canal que o artefato ja tem para tempo decomposto, e o
            # `metric_series` do `PipelineResult` ja o emite como `stage_<nome>_seconds`.
            result.stage_seconds["service_p50_ms"] = sorted(servicos)[len(servicos) // 2]
            result.stage_seconds["response_p50_ms"] = sorted(respostas)[len(respostas) // 2]
        # Qualidade NAO e reportada sob concorrencia: as consultas dao a volta, entao um nDCG medio
        # seria sobre um conjunto repetido e nao sobre o conjunto julgado. A pergunta aqui e vazao.
        return result

    def run_pipeline(
        self, adapter: SystemAdapter, pipeline: str, repetition: int
    ) -> PipelineResult:
        """One timed pass of one pipeline over the whole query set."""
        result = PipelineResult(pipeline=pipeline, repetition=repetition)

        capability = PIPELINE_CAPABILITY[pipeline]
        if not adapter.supports(capability):
            result.status = "unsupported"
            result.status_detail = f"{adapter.system_id} does not support {capability}"
            return result

        latencies: list[float] = []
        ndcgs: list[float] = []
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        stages: dict[str, float] = {}

        started = time.perf_counter()
        for index in range(len(self.queries.texts)):
            try:
                ranked, elapsed, stage_seconds = self._run_one(adapter, pipeline, index)
            except SystemUnavailableError:
                raise
            except MeasurementError:
                result.timeouts += 1
                continue
            except UnsupportedCapabilityError as exc:
                result.status = "unsupported"
                result.status_detail = exc.message
                return result
            except BenchError:
                result.errors += 1
                continue

            latencies.append(elapsed * 1000.0)
            result.latency_by_query[index] = elapsed * 1000.0
            for stage, seconds in stage_seconds.items():
                stages[stage] = stages.get(stage, 0.0) + seconds

            judgements = self.queries.relevance[index]
            relevant = self.queries.relevant_ids(index)
            ndcg_desta = ndcg_at_k(list(ranked), judgements, 10)
            ndcgs.append(ndcg_desta)
            result.quality_by_query[index] = ndcg_desta
            recalls.append(recall_at_n(list(ranked), relevant, self.workload.k))
            reciprocal_ranks.append(mrr_at_k(list(ranked), relevant, self.workload.k))
        result.duration_seconds = time.perf_counter() - started

        result.successes = len(latencies)
        result.latency = summarise_latency(latencies)
        result.stage_seconds = stages
        # None rather than 0.0 when nothing came back: a pipeline that answered
        # nothing has no measured quality, and zero would read as terrible
        # quality instead of an absence of measurement.
        result.ndcg_at_10 = float(np.mean(ndcgs)) if ndcgs else None
        result.recall_at_k = float(np.mean(recalls)) if recalls else None
        result.mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        return result

    # ------------------------------------------------------------------ stages

    def _run_one(
        self, adapter: SystemAdapter, pipeline: str, index: int
    ) -> tuple[Sequence[int], float, dict[str, float]]:
        text = self.queries.texts[index]
        vector = self.queries.vectors[index]
        table = self.workload.table
        n = self.workload.n

        if pipeline == LEXICAL:
            outcome = adapter.execute_lexical(LexicalQuery(table, text, n))
            return outcome.ids, outcome.latency_seconds, dict(outcome.stage_seconds)

        if pipeline == VECTOR:
            knn = adapter.execute(
                KnnQuery(table=table, vector=vector, k=n, metric=self.workload.metric)
            )
            return knn.ids, knn.latency_seconds, {"vector": knn.latency_seconds}

        if pipeline == HYBRID_RRF:
            outcome = adapter.execute_hybrid(
                HybridQuery(
                    table=table,
                    text=text,
                    vector=vector,
                    n=n,
                    metric=self.workload.metric,
                    rrf_k=self.workload.rrf_k,
                )
            )
            return outcome.ids, outcome.latency_seconds, dict(outcome.stage_seconds)

        # hybrid + rerank: fuse first, then reorder the candidates. The model's
        # time arrives in its own stage and is never added to the database's.
        fused = adapter.execute_hybrid(
            HybridQuery(
                table=table,
                text=text,
                vector=vector,
                n=n,
                metric=self.workload.metric,
                rrf_k=self.workload.rrf_k,
            )
        )
        reranked = adapter.execute_rerank(text, list(fused.ids))
        stages = dict(fused.stage_seconds)
        for stage, seconds in reranked.stage_seconds.items():
            stages[f"rerank_{stage}"] = stages.get(f"rerank_{stage}", 0.0) + seconds
        return (
            reranked.ids,
            fused.latency_seconds + reranked.latency_seconds,
            stages,
        )

    # ------------------------------------------------------------- offline twin

    def offline_fusion(self, lexical_ids: Sequence[int], vector_ids: Sequence[int]) -> list[int]:
        """Fuse two legs here, so the system's own fusion can be checked."""
        return fuse_to_ids(
            {"lexical": list(lexical_ids), "vector": list(vector_ids)},
            n=self.workload.n,
            k=self.workload.rrf_k,
        )

    def points(
        self,
        adapter: SystemAdapter,
        repetitions: int,
        make_client: Callable[[], Any] | None = None,
    ) -> list[RetrievalPoint]:
        """Um ponto por (pipeline, clientes), com uma repeticao por passada.

        Sem `client_sweep` e um cliente so, em serie — que e o que as suites existentes usam.

        Com varredura (B-043), cada contagem de clientes vira um ponto proprio, em laco fechado.
        E isso que distingue "o trabalho por consulta e caro" de "ha fila contra capacidade fixa":
        a segunda so aparece com mais de um cliente, e era justamente o que o arnes nao sabia
        produzir.
        """
        pontos: list[RetrievalPoint] = []
        contagens = self.workload.client_sweep or (1,)
        for pipeline in self.workload.pipelines:
            self.warm_up(adapter, pipeline)
            for n_clientes in contagens:
                # 1-based: o schema do artefato exige `repetition >= 1`. `range(repetitions)` daria
                # zero na primeira, e o validador recusa — corretamente, porque "repeticao 0" nao
                # significa nada para quem le o artefato depois.
                faixa = range(1, repetitions + 1)
                if n_clientes == 1:
                    passadas = [self.run_pipeline(adapter, pipeline, r) for r in faixa]
                else:
                    passadas = [
                        self.run_concurrent(adapter, pipeline, r, n_clientes, make_client)
                        for r in faixa
                    ]
                # O status do ponto e o da primeira passada: `unsupported` e propriedade do adapter
                # e nao varia entre repeticoes. Um pipeline sem suporte reporta isso UMA vez, em vez
                # de somar tres repeticoes vazias que pareceriam medicao.
                estado = passadas[0].status if passadas else "unsupported"
                sufixo = f" @ {n_clientes} clientes" if self.workload.client_sweep else ""
                pontos.append(
                    RetrievalPoint(
                        label=f"pipeline={pipeline}{sufixo}",
                        parameters={
                            "pipeline": pipeline,
                            "k": self.workload.k,
                            "n": self.workload.n,
                            "clients": n_clientes,
                        },
                        status=estado,
                        status_detail=passadas[0].status_detail if passadas else None,
                        repetitions=passadas,
                    )
                )
        return pontos

    def summary(self, results: Sequence[PipelineResult]) -> dict[str, Any]:
        """A comparison across pipelines on the same corpus and query set."""
        return {
            "corpus_size": self.workload.corpus_size,
            "query_count": self.workload.query_count,
            "k": self.workload.k,
            "candidate_depth": self.workload.n,
            "pipelines": {
                result.pipeline: {
                    "status": result.status,
                    "ndcg_at_10": result.ndcg_at_10,
                    "recall_at_k": result.recall_at_k,
                    "mrr": result.mrr,
                    "throughput_per_second": result.throughput,
                    "stage_seconds": result.stage_seconds,
                }
                for result in results
            },
            # `paired_quality` NÃO entra aqui, e a razão é o motivo de este comentário existir.
            #
            # `summary()` tem **zero chamadores**: o runner pede ao benchmark apenas `load` e
            # `points`, e o protocolo não tem espaço para um resultado que compara CAMINHOS em
            # vez de descrever um ponto. Pôr o veredito aqui o tornaria invisível — que é
            # exatamente o defeito que `assert_analytical_path` teve, e que este arnês já pagou
            # três vezes em 2026-08-22.
            #
            # `veredito_de_qualidade` e `quality_by_query` existem, estão testados e servem a quem
            # os chame. Dar-lhes uma casa que RODE exige mexer no protocolo, e isso é trabalho
            # com desenho próprio — registrado, não improvisado.
        }
