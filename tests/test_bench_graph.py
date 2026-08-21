"""A wrong traversal is not a fast traversal. Timings follow validation."""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import TraversalQuery
from theodb_bench.adapters.fake import FakeAdapter, FakeConfig
from theodb_bench.bench.graph import (
    BUILD,
    FANOUT_SWEEP,
    NEIGHBOURHOOD,
    ONE_HOP,
    REBUILD,
    THREE_HOP,
    TWO_HOP,
    GraphBenchmark,
    GraphWorkload,
    build_adjacency,
    generate_graph,
    rebuild_delta,
    timed_reference_traversal,
    true_neighbourhood,
)
from theodb_bench.errors import AdapterError, BenchError, ConfigError


def _workload(**overrides: object) -> GraphWorkload:
    base: dict[str, object] = {"vertex_count": 200, "average_degree": 4, "query_count": 20}
    base.update(overrides)
    return GraphWorkload(**base)  # type: ignore[arg-type]


def _ready(config: FakeConfig | None = None) -> FakeAdapter:
    adapter = FakeAdapter(config)
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()
    return adapter


def _loaded(workload: GraphWorkload) -> tuple[FakeAdapter, GraphBenchmark]:
    adapter = _ready()
    benchmark = GraphBenchmark(workload)
    benchmark.load(adapter)
    return adapter, benchmark


# ------------------------------------------------------------------- workload


def test_unknown_workloads_are_refused() -> None:
    with pytest.raises(ConfigError, match="unknown graph workload"):
        _workload(workloads=("teleport",))


def test_a_degenerate_graph_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least 2 vertices"):
        _workload(vertex_count=1)
    with pytest.raises(ConfigError, match="average degree"):
        _workload(average_degree=0)


def test_the_graph_is_deterministic_for_a_seed() -> None:
    assert generate_graph(_workload(seed=4)) == generate_graph(_workload(seed=4))


def test_a_different_seed_gives_a_different_graph() -> None:
    assert generate_graph(_workload(seed=4)) != generate_graph(_workload(seed=5))


# --------------------------------------------------------------------- oracle


def test_the_oracle_expands_one_hop_correctly() -> None:
    adjacency = {0: [1, 2], 1: [3], 2: [], 3: []}
    assert true_neighbourhood(adjacency, 0, 1) == [0, 1, 2]


def test_the_oracle_expands_two_hops_without_revisiting() -> None:
    adjacency = {0: [1, 2], 1: [3], 2: [3], 3: []}
    assert true_neighbourhood(adjacency, 0, 2) == [0, 1, 2, 3]


def test_the_oracle_terminates_on_a_cycle() -> None:
    adjacency = {0: [1], 1: [2], 2: [0]}
    assert true_neighbourhood(adjacency, 0, 10) == [0, 1, 2]


def test_the_oracle_includes_the_source_because_the_system_does() -> None:
    """SUBSTITUI `test_the_oracle_excludes_the_source`, que codificava o contrato oposto.

    O teste anterior nao estava errado sobre o codigo — estava errado sobre o sistema. Ele
    afirmava que o oraculo exclui a fonte, e o oraculo de fato excluia; o que ninguem havia
    conferido e que o CSR medido a **inclui**. Um teste verde sobre uma definicao que o sistema
    sob medicao nao usa e o mecanismo pelo qual isto sobreviveu ate 2026-08-21.
    """
    adjacency = {0: [1], 1: [0]}
    assert true_neighbourhood(adjacency, 0, 1) == [0, 1]


def test_an_undirected_graph_has_symmetric_adjacency() -> None:
    adjacency = build_adjacency([(0, 1)], 2, directed=False)
    assert adjacency[1] == [0]
    directed = build_adjacency([(0, 1)], 2, directed=True)
    assert directed[1] == []


# ------------------------------------------------------------------ traversal


@pytest.mark.parametrize("name", [ONE_HOP, TWO_HOP, THREE_HOP])
def test_traversal_agrees_with_the_oracle_and_is_timed(name: str) -> None:
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run(adapter, name)
        assert result.status == "measured"
        assert result.incorrect_traversals == 0
        assert result.queries == workload.query_count
        assert result.edges_per_second is not None
        assert result.nanoseconds_per_edge is not None
    finally:
        adapter.stop()


def test_more_hops_visit_more_edges() -> None:
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        one = benchmark.run(adapter, ONE_HOP)
        three = benchmark.run(adapter, THREE_HOP)
        assert three.edges_visited > one.edges_visited
    finally:
        adapter.stop()


def test_edges_visited_measures_work_not_answer_size() -> None:
    # A traversal returning few vertices after walking many edges is expensive,
    # and the metric has to show that.
    workload = _workload()
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run(adapter, TWO_HOP)
        assert result.edges_visited > 0
        assert result.nanoseconds_per_edge is not None and result.nanoseconds_per_edge > 0
    finally:
        adapter.stop()


def test_a_wrong_traversal_discards_its_timing() -> None:
    # The defect this prevents: a broken implementation ranking first because
    # returning the wrong answer is quick.
    workload = _workload(query_count=5)
    adapter, benchmark = _loaded(workload)
    try:
        # Corrupt the oracle so every system answer disagrees with it.
        benchmark.adjacency = {v: [] for v in range(workload.vertex_count)}
        result = benchmark.run(adapter, TWO_HOP)
        assert result.status == "invalid"
        assert result.incorrect_traversals == 5
        assert result.queries == 0
        assert result.status_detail is not None
        assert "wrong answer is not a fast one" in result.status_detail
    finally:
        adapter.stop()


def test_traversing_from_a_missing_vertex_is_an_error() -> None:
    adapter, _ = _loaded(_workload())
    try:
        with pytest.raises(AdapterError, match="not in the graph"):
            adapter.traverse(TraversalQuery(graph="bench_graph", source=10**6, hops=1))
    finally:
        adapter.stop()


# ------------------------------------------------------- bounded neighbourhood


def test_a_bounded_expansion_honours_its_limit() -> None:
    workload = _workload(neighbourhood_limit=5)
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run(adapter, NEIGHBOURHOOD)
        assert result.status == "measured"
        assert result.queries == workload.query_count
    finally:
        adapter.stop()


def test_a_capped_walk_still_counts_the_edges_it_walked() -> None:
    # Stopping early does not make the work disappear.
    adapter, _ = _loaded(_workload())
    try:
        capped = adapter.traverse(TraversalQuery(graph="bench_graph", source=0, hops=3, limit=2))
        assert len(capped.vertices) <= 2
        assert capped.edges_visited > 0
    finally:
        adapter.stop()


# --------------------------------------------------------------------- fanout


def test_the_fanout_sweep_reports_cost_per_edge_at_each_degree() -> None:
    workload = _workload(vertex_count=150, fanout_degrees=(2, 8), query_count=10)
    adapter, benchmark = _loaded(workload)
    try:
        result = benchmark.run(adapter, FANOUT_SWEEP)
        assert set(result.fanout) == {2, 8}
        assert all(value > 0 for value in result.fanout.values())
    finally:
        adapter.stop()


def test_the_sweep_restores_the_declared_graph() -> None:
    # A later workload must measure the graph it declared, not the last one
    # the sweep happened to build.
    workload = _workload(vertex_count=150, fanout_degrees=(2, 32), query_count=10)
    adapter, benchmark = _loaded(workload)
    try:
        benchmark.run(adapter, FANOUT_SWEEP)
        after = benchmark.run(adapter, TWO_HOP)
        assert after.incorrect_traversals == 0
    finally:
        adapter.stop()


# ---------------------------------------------------------------------- build


def test_build_is_timed_and_sized_apart_from_queries() -> None:
    adapter = _ready()
    try:
        result = GraphBenchmark(_workload()).run(adapter, BUILD)
        assert result.build_seconds is not None
        assert result.structure_bytes is not None and result.structure_bytes > 0
        assert result.bytes_per_edge is not None
    finally:
        adapter.stop()


def test_rebuild_is_comparable_with_the_first_build() -> None:
    adapter = _ready()
    try:
        benchmark = GraphBenchmark(_workload())
        first = benchmark.run(adapter, BUILD)
        second = benchmark.run(adapter, REBUILD)
        delta = rebuild_delta(first, second)
        assert delta["delta_seconds"] is not None
        assert "would not know which number it had" in delta["note"]
    finally:
        adapter.stop()


def test_comparing_builds_refuses_when_one_was_not_measured() -> None:
    from theodb_bench.bench.graph import GraphResult

    delta = rebuild_delta(GraphResult(workload=BUILD), GraphResult(workload=REBUILD))
    assert delta["delta_seconds"] is None
    assert "both builds must be measured" in delta["note"]


# ------------------------------------------------------------------ reference


def test_the_reference_walk_gives_a_floor_to_read_against() -> None:
    workload = _workload()
    edges = generate_graph(workload)
    adjacency = build_adjacency(edges, workload.vertex_count, directed=True)
    seconds, walked = timed_reference_traversal(adjacency, list(range(10)), hops=2)
    assert seconds > 0
    assert walked > 0


# ---------------------------------------------------------------- unsupported


def test_a_system_without_graph_support_reports_unsupported() -> None:
    adapter = _ready(FakeConfig(capabilities={"vector_exact": True}))
    try:
        result = GraphBenchmark(_workload()).run(adapter, TWO_HOP)
        assert result.status == "unsupported"
        assert result.queries == 0
    finally:
        adapter.stop()


def test_metric_series_expose_the_work_units() -> None:
    adapter, benchmark = _loaded(_workload())
    try:
        series = benchmark.run(adapter, TWO_HOP).metric_series()
        assert "edges_per_second" in series
        assert "nanoseconds_per_edge" in series
    finally:
        adapter.stop()


# ---------------------------------------------------------------------------
# B-007 — os dois lados da comparacao precisam medir a MESMA coisa.
#
# Medido em 2026-08-21, contra um TheoDB real, e o achado quase virou um numero publicado:
# para a fonte 1048, o oraculo dizia 8 vertices, o `WITH RECURSIVE` devolvia os mesmos 8, e o CSR
# devolvia 22 — a vizinhanca NAO-dirigida mais a propria fonte. Cronometrar uma expansao de 22
# contra uma de 8 e chamar a razao de "speedup" teria sido a mesma classe de defeito que a
# retratacao lexical desta sessao: um numero honesto sobre uma comparacao que nao existia.
#
# O CSR nao tem defeito: `theodb_rs/src/graph.rs:11` documenta *"undirected, <=H hops"* e `:429`
# fala em **reachable set**, que inclui a semente (alcancavel em 0 saltos). Quem estava errado era
# o arnes, em quatro pontos — um teste por ponto.


def test_oraculo_inclui_a_semente_porque_o_sistema_medido_a_inclui() -> None:
    """`reachable set` inclui a propria semente; o oraculo tem de modelar isso."""
    adjacency = {0: [1, 2], 1: [0], 2: [0], 3: []}
    assert 0 in true_neighbourhood(adjacency, 0, 1)


def test_workload_de_grafo_nao_oferece_knob_dirigido() -> None:
    """Um knob aceito sem efeito e pior que knob nenhum.

    `GraphSpec.directed` so era lido pelo adapter fake; o `PostgresAdapter` nunca o leu, e a
    extensao so tem CSR nao-dirigido. Declarar `directed=True` e ver a medicao rodar dava a
    impressao de que a direcao fora respeitada.
    """
    assert not hasattr(GraphWorkload(vertex_count=8, average_degree=2), "directed")


def test_adapter_real_recusa_grafo_dirigido_em_vez_de_ignorar_o_pedido() -> None:
    from theodb_bench.adapters.base import GraphSpec
    from theodb_bench.adapters.postgres import PostgresConfig, TheoDBAdapter

    adapter = TheoDBAdapter(PostgresConfig(dsn="postgresql://x/y"))
    with pytest.raises(BenchError, match="dirigid"):
        adapter.load_graph(GraphSpec(name="g", directed=True), [(0, 1)], 2)


def test_corretude_de_travessia_compara_conjunto_e_nao_ordem() -> None:
    """A semantica comparada e "vertices alcancados" — um conjunto.

    O `WITH RECURSIVE` devolve na ordem que o planner escolher, e o CSR na ordem da fronteira.
    Exigir ordem reprovava um baseline que estava certo.
    """
    workload = GraphWorkload(vertex_count=400, average_degree=8, query_count=1)
    benchmark = workload.build(None, None)
    source = benchmark.sources[0]
    esperado = true_neighbourhood(benchmark.adjacency, source, 1)
    # Sem isto o teste passa por vacuidade: uma vizinhanca de um elemento e igual a si mesma
    # invertida, e a asercao abaixo nao exercitaria ordem nenhuma.
    assert len(esperado) >= 2, "a fonte escolhida precisa ter vizinhanca suficiente para reordenar"
    assert benchmark._is_correct(source, 1, None, list(reversed(esperado)))


@pytest.mark.integration
def test_csr_e_sql_recursivo_concordam_com_o_mesmo_oraculo() -> None:
    """A propriedade que faz a comparacao do [[B-007]] significar alguma coisa.

    Exige um TheoDB real (`PGHOST`/`PGPORT`/`PGUSER`), porque o defeito que ela existe para pegar
    e invisivel contra o fake: enquanto o fake excluia a fonte, os testes unitarios ficavam verdes
    e o servidor discordava do oraculo em 100% das travessias.

    Nao afirma que os dois sao igualmente rapidos — afirma que respondem **a mesma pergunta**.
    Sem isso, a razao entre os dois tempos e um numero sem referente.
    """
    import os

    if not os.environ.get("PGPORT"):
        pytest.skip("sem servidor declarado: PGPORT ausente")

    from theodb_bench.adapters.postgres import PostgresConfig, TheoDBAdapter

    dsn = (
        f"postgresql://{os.environ.get('PGUSER', 'postgres')}"
        f"@{os.environ.get('PGHOST', '127.0.0.1')}:{os.environ['PGPORT']}/postgres"
    )
    adapter = TheoDBAdapter(PostgresConfig(dsn=dsn))
    adapter.prepare()
    adapter.start()
    adapter.wait_ready()

    workload = GraphWorkload(vertex_count=2_000, average_degree=8, query_count=5)
    benchmark = workload.build(None, None)
    benchmark.load(adapter)

    for source in benchmark.sources[:3]:
        for hops in (1, 2, 3):
            # Confere pelo MESMO criterio do portao (`_is_correct`), e nao por `set(...)`.
            # Uma versao anterior deste teste comparava so conjuntos e deu por corretos dois
            # baselines que devolviam vertices REPETIDOS — o `UNION` do CTE deduplica a linha
            # `(v, salto)`, nao o vertice. Um teste mais frouxo que o portao que ele protege nao
            # protege nada.
            csr = adapter.traverse(
                TraversalQuery(graph=workload.graph, source=source, hops=hops, limit=None)
            ).vertices
            sql = adapter.traverse_recursive_sql(
                TraversalQuery(graph=workload.graph, source=source, hops=hops)
            ).vertices
            assert benchmark._is_correct(source, hops, None, csr), (
                f"CSR discordou do oraculo em {source}/{hops} saltos"
            )
            assert benchmark._is_correct(source, hops, None, sql), (
                f"SQL recursivo discordou do oraculo em {source}/{hops} saltos"
            )
