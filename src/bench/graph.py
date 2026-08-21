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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from theodb_bench.adapters.base import GraphSpec, SystemAdapter, TraversalQuery
from theodb_bench.analysis.statistics import LatencySummary, summarise_latency
from theodb_bench.errors import BenchError, ConfigError, ErrorContext, Phase

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

    #: Comparar contra `WITH RECURSIVE` no mesmo servidor — o baseline que o [[B-007]] pede:
    #: *"SQL recursivo no proprio Postgres serve, e e o que o usuario faria sem nos"*.
    #:
    #: Distinto do `timed_reference_traversal`, que e um passeio em dicionario na memoria e serve
    #: de PISO. Este e uma ALTERNATIVA REAL: mesmo servidor, mesmos dados, mesmo MVCC.
    compare_recursive_sql: bool = False

    def spec(self) -> GraphSpec:
        # `directed=False` fixo, e nao um knob: a extensao so tem CSR nao-dirigido
        # (`theodb_rs/src/graph.rs:44`). Ver o teste que barra o knob de voltar.
        return GraphSpec(name=self.graph, directed=False)

    # ---- protocolo `Workload` (bench/protocol.py) ----
    #
    # Faltavam, e a consequencia esta medida: `bench.graph` estava na lista de orfaos de
    # `tests/test_module_reachability.py` — 23 funcoes de grafo expostas no binario e **nenhum
    # benchmark registrado as alcancava**, que e literalmente o que o [[B-007]] registra.

    def build(self, corpus: Any, queries: Any) -> GraphBenchmark:
        """O grafo e semeado a partir do proprio workload; `corpus`/`queries` nao se aplicam."""
        del corpus, queries
        return GraphBenchmark(self)

    def benchmark_payload(self) -> dict[str, Any]:
        return {
            "workload": {
                "type": "graph",
                "loop": "closed",
                "clients": 1,
                "arrival_rate_per_second": None,
                "operation_count": self.query_count,
            },
            # A travessia e certa ou errada contra o vizinhanca exata computada aqui — nao ha
            # aproximacao a pontuar. `exact_match` e o termo do vocabulario para isso.
            "quality": {"metric": "exact_match", "ground_truth": "computed"},
            "parameters": {
                "workloads": list(self.workloads),
                "average_degree": [self.average_degree],
                **({"baseline": ["recursive_sql"]} if self.compare_recursive_sql else {}),
            },
        }

    def expected_operations(self, measured_points: int, repetitions: int) -> int:
        return measured_points * repetitions * self.query_count

    @property
    def warmup_operations(self) -> int:
        """Uma travessia por fonte, descartada, antes de cada ponto medido.

        MEDIDO em 2026-08-21, e o numero muda a conclusao. Na primeira corrida deste benchmark o
        p50 de 1 salto (0,731 ms) saiu MAIOR que o de 2 saltos (0,439 ms) — mais trabalho custando
        menos, que nao pode ser verdade. Invertendo a ordem dos workloads, o efeito seguiu a ORDEM
        e nao o workload: quem roda primeiro paga. A quente, 1 salto cai para ~0,19 ms.

        A conta que isso desfaz: sem aquecimento, o SQL recursivo parecia 6,2x mais rapido que o
        CSR a 1 salto; a quente, sao ~2,2x. O resultado continua desfavoravel para nos — e agora
        e o numero certo. Publicar 6,2x teria sido publicar o custo da primeira chamada.
        """
        return self.query_count

    def quality_was_reported(self, points: list[Any]) -> bool:
        # A corretude e um veredito por ponto (`status`), nao um numero por repeticao: uma
        # travessia errada NAO tem timing aproveitavel, e o benchmark ja a descarta.
        return bool(points)


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
    """O conjunto alcancavel em ate `hops` saltos, **incluindo a propria fonte**.

    Calculado aqui, das mesmas arestas que o sistema recebeu. E o oraculo contra o qual toda
    travessia e conferida, e e por isso que uma resposta errada nao pode ser reportada como rapida.

    **A fonte entra no conjunto**, e isso nao e detalhe: a semantica sob medicao e a que
    `theodb_rs/src/graph.rs:429` documenta — *reachable set* dentro de <=H saltos — e a semente e
    alcancavel em zero saltos. Ate 2026-08-21 o oraculo a excluia e o sistema a incluia, de modo
    que **toda** travessia era reprovada por discordancia. Duas definicoes defensaveis; comparar
    uma com a outra e que nao era.
    """
    seen = {source}
    reached: list[int] = [source]
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
    repetition: int = 1
    """Qual repeticao produziu este resultado. O runner serializa as latencias por consulta
    indexadas por repeticao, e sem este campo o bundle nao se escreve."""

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

    recall: float | None = None
    """SEMPRE `None`, e de proposito. A qualidade de uma travessia e *exact match* contra o
    oraculo — a resposta bate ou o tempo dela e descartado (ver `incorrect_traversals`) — e
    `recall@k` nao tem referente aqui. Preencher com `1.0` faria o relatorio publicar uma
    metrica de qualidade que ninguem mediu; `None` faz ele pular o eixo, que e o correto."""

    timeouts: int = 0
    errors: int = 0
    successes: int = 0
    duration_seconds: float | None = None
    throughput: float | None = None
    """Os contadores que o runner agrega sobre TODA corrida, seja qual for o pilar. Este benchmark
    e de latencia em laco fechado com um cliente so: `throughput` fica `None` em vez de virar
    `queries/duracao`, porque vazao com um cliente mede o round-trip, nao a capacidade do servidor
    — foi exatamente essa confusao que produziu a retratacao lexical de 2026-08-20."""

    index_size_bytes: int | None = None
    """Alias de `structure_bytes` para o contrato do runner: o mesmo numero, o nome que ele le."""

    latency_by_query: dict[int, float] = field(default_factory=dict)
    """Latencia por fonte, em ms. E o unico lugar onde o valor por consulta existe, e e o que
    permite ao `compare` parear duas corridas e rodar o teste pareado que a I14 exige — um resumo
    nao se pareia. A chave e o indice da fonte na lista semeada, nao o id do vertice: e o indice
    que duas corridas com a mesma semente compartilham."""

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


def _status_de_artefato(status: str) -> str:
    """Traduz o vocabulario interno para o que o schema do artefato aceita.

    `invalid` diz por que o resultado nao vale — a travessia discordou do oraculo — e essa razao
    fica em `status_detail`. O schema so conhece quatro estados, e `failed` e o que significa a
    mesma coisa la: nao ha numero para ler. A traducao acontece **na fronteira**, para que o
    diagnostico nao se perca dentro do arnes.
    """
    return "failed" if status == "invalid" else status


def _numeradas(passadas: Iterable[GraphResult]) -> list[GraphResult]:
    """Numera as repeticoes a partir de 1.

    Feito aqui e nao dentro de cada `run` porque quem sabe qual repeticao esta correndo e o laco,
    nao a travessia. O runner indexa as latencias por consulta por este numero.
    """
    resultados = list(passadas)
    for numero, resultado in enumerate(resultados, start=1):
        resultado.repetition = numero
    return resultados


class GraphBenchmark:
    """Runs graph workloads and validates every traversal before timing it."""

    def __init__(self, workload: GraphWorkload) -> None:
        self.workload = workload
        self.edges = generate_graph(workload)
        self.adjacency = build_adjacency(self.edges, workload.vertex_count, directed=False)
        rng = np.random.default_rng(workload.seed + 1)
        self.sources = [
            int(v) for v in rng.integers(0, workload.vertex_count, size=workload.query_count)
        ]

    def load(self, adapter: SystemAdapter) -> float | None:
        """Constroi a estrutura e devolve os segundos — a assinatura que o `Benchmark` pede.

        Fina de proposito. A construcao rende uma medicao rica (bytes por aresta, arestas
        contadas), e o protocolo so quer um numero; devolver o objeto inteiro daqui fazia
        `GraphBenchmark` deixar de satisfazer `Benchmark`, que foi o que o mypy apanhou. A medicao
        completa continua acessivel por `build_measurement`, e e o que o workload `build` reporta.
        """
        return self.build_measurement(adapter).build_seconds

    def build_measurement(self, adapter: SystemAdapter) -> GraphResult:
        """Build the structure, timed apart from any query."""
        result = GraphResult(workload=BUILD)
        outcome = adapter.load_graph(self.workload.spec(), self.edges, self.workload.vertex_count)
        result.build_seconds = outcome.seconds
        result.structure_bytes = outcome.index_size_bytes
        result.index_size_bytes = outcome.index_size_bytes
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
            return self.build_measurement(adapter)
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

        # Aquecimento descartado — ver `GraphWorkload.warmup_operations` para a medicao que
        # obriga a isto. O CSR e carregado sob demanda, e a primeira travessia paga por todas.
        for source in self.sources:
            adapter.traverse(
                TraversalQuery(graph=self.workload.graph, source=source, hops=hops, limit=limit)
            )

        for indice, source in enumerate(self.sources):
            outcome = adapter.traverse(
                TraversalQuery(graph=self.workload.graph, source=source, hops=hops, limit=limit)
            )
            if not self._is_correct(source, hops, limit, outcome.vertices):
                # The timing that accompanied a wrong answer is not evidence
                # about traversal speed.
                result.incorrect_traversals += 1
                continue
            result.latency_by_query[indice] = outcome.latency_seconds * 1000.0
            latencies.append(outcome.latency_seconds * 1000.0)
            total_edges += outcome.edges_visited
            total_seconds += outcome.latency_seconds

        result.queries = len(latencies)
        result.latency = summarise_latency(latencies)
        result.edges_visited = total_edges
        result.successes = len(latencies)
        result.errors = result.incorrect_traversals
        result.duration_seconds = total_seconds
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

    def _baseline_recursive_sql(
        self, adapter: SystemAdapter, name: str, *, hops: int
    ) -> GraphResult:
        """A MESMA travessia por `WITH RECURSIVE` — o baseline do [[B-007]].

        Roda sobre a mesma tabela de arestas, no mesmo servidor, pagando o mesmo MVCC. E a
        pergunta que o usuario tem: *vale a pena instalar isto em vez de escrever um
        `WITH RECURSIVE`?*

        A corretude e conferida contra o MESMO oraculo. Um baseline que devolvesse a resposta
        errada rapido nao seria um baseline — e essa checagem ja apanhou o caso de `limit`, que o
        SQL recursivo nao implementa e por isso nao e comparado.
        """
        result = GraphResult(workload=f"{name}/recursive_sql")
        latencies: list[float] = []
        total_edges = 0
        total_seconds = 0.0

        # O mesmo aquecimento do outro lado. Dar aquecimento a um so e escolher o vencedor: a
        # primeira consulta paga o buffer cache do indice e da tabela de arestas.
        for source in self.sources:
            try:
                adapter.traverse_recursive_sql(
                    TraversalQuery(graph=self.workload.graph, source=source, hops=hops)
                )
            except BenchError as exc:
                result.status = "unsupported"
                result.status_detail = str(exc)
                return result

        for indice, source in enumerate(self.sources):
            try:
                outcome = adapter.traverse_recursive_sql(
                    TraversalQuery(graph=self.workload.graph, source=source, hops=hops)
                )
            except BenchError as exc:
                result.status = "unsupported"
                result.status_detail = str(exc)
                return result
            if not self._is_correct(source, hops, None, outcome.vertices):
                result.incorrect_traversals += 1
                continue
            result.latency_by_query[indice] = outcome.latency_seconds * 1000.0
            latencies.append(outcome.latency_seconds * 1000.0)
            total_edges += outcome.edges_visited
            total_seconds += outcome.latency_seconds
        result.queries = len(latencies)
        result.latency = summarise_latency(latencies)
        result.edges_visited = total_edges
        result.successes = len(latencies)
        result.errors = result.incorrect_traversals
        result.duration_seconds = total_seconds
        if total_seconds > 0 and total_edges > 0:
            result.edges_per_second = total_edges / total_seconds
            result.nanoseconds_per_edge = total_seconds * 1e9 / total_edges
        if result.incorrect_traversals:
            result.status = "invalid"
            result.status_detail = (
                f"{result.incorrect_traversals} travessia(s) do baseline discordaram do oraculo"
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
            # Conjunto e nao lista: a semantica comparada e "vertices alcancados", e nem o
            # `WITH RECURSIVE` (ordem do planner) nem o CSR (ordem da fronteira) prometem ordem.
            # Duplicata continua sendo erro, e por isso o tamanho e conferido junto.
            return len(returned) == len(expected) and set(returned) == set(expected)
        expected_set = set(expected)
        return len(returned) == min(limit, len(expected)) and set(returned) <= expected_set

    def points(
        self,
        adapter: SystemAdapter,
        repetitions: int,
        make_client: Callable[[], SystemAdapter] | None = None,
    ) -> list[GraphPoint]:
        """Um ponto por workload declarado — e, quando pedido, um ponto irmao para o baseline.

        `make_client` e aceito e ignorado: esta familia emite trabalho em serie, e o protocolo diz
        que um benchmark serial deve ACEITAR o argumento em vez de o runner ter de saber qual tipo
        segura.

        O baseline vira PONTO PROPRIO e nao um campo dentro do ponto do CSR. Dois pontos com o
        mesmo `parameters.workload` e `engine` diferente e o que deixa o artefato comparavel pelas
        mesmas ferramentas que comparam dois sistemas — dobrar um dentro do outro pediria um leitor
        especial.
        """
        del make_client
        pontos: list[GraphPoint] = []
        for nome in self.workload.workloads:
            passadas = _numeradas(self.run(adapter, nome) for _ in range(max(1, repetitions)))
            pontos.append(
                GraphPoint(
                    label=f"graph={nome}",
                    parameters={"workload": nome, "engine": "csr"},
                    status=_status_de_artefato(passadas[0].status),
                    status_detail=passadas[0].status_detail,
                    repetitions=passadas,
                )
            )
            if not self.workload.compare_recursive_sql or nome not in _HOPS:
                continue
            base = _numeradas(
                self._baseline_recursive_sql(adapter, nome, hops=_HOPS[nome])
                for _ in range(max(1, repetitions))
            )
            pontos.append(
                GraphPoint(
                    label=f"graph={nome} via recursive_sql",
                    parameters={"workload": nome, "engine": "recursive_sql"},
                    status=_status_de_artefato(base[0].status),
                    status_detail=base[0].status_detail,
                    repetitions=base,
                )
            )
        return pontos

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
            adjacency = build_adjacency(edges, self.workload.vertex_count, directed=False)
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


@dataclass
class GraphPoint:
    """Uma configuracao medida — aqui, um workload de grafo — com suas repeticoes.

    Mesma forma que o runner le nas outras familias: `label`, `parameters`, `status`,
    `repetitions` e `metric_series()`.
    """

    label: str
    parameters: dict[str, Any]
    status: str = "measured"
    status_detail: str | None = None
    repetitions: list[GraphResult] = field(default_factory=list)

    def metric_series(self) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        for r in self.repetitions:
            for nome, valores in r.metric_series().items():
                series.setdefault(nome, []).extend(valores)
        return series


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
