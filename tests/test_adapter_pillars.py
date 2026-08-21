"""The surfaces beyond vector, and the ones that stay honestly unreachable.

Measured 2026-08-17 by crossing `CAPABILITIES` with what every registered adapter
declares: six of fourteen capabilities were reachable and eight had no adapter at
all. The scaffolding for three of them existed and was disconnected.

What TheoDB actually exposes, read from `pg_proc` on the running server rather
than from documentation — and it is not where the docs implied:

    public.bm25_build(index_id bigint, table text, id_col text, text_col text)
    public.bm25_search(index_id bigint, query text, k integer)
    public.read_parquet(path text) -> SETOF jsonb
    public.write_parquet(rel text, path text) -> bigint
    ai.hybrid_search_rrf(tbl regclass, id_col, content_tsv_col, vector_col, ...)
    theodb.graph_build(edge_rel text, src_col text, dst_col text)
    theodb.graph_expand(edge_rel text, seeds bigint[], max_hops integer)

Three capabilities stay undeclared and that is the finding, not the gap: `rerank`,
`vectorizer` and `ai_sql` all reach an external model. Without an endpoint there
is nothing to measure, and a stub would put a number where an absence belongs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from theodb_bench.adapters.base import (
    CAPABILITIES,
    AnalyticalTable,
    Document,
    DocumentTableSpec,
    GraphSpec,
    HybridQuery,
    LexicalQuery,
    TraversalQuery,
)
from theodb_bench.adapters.postgres import TheoDBAdapter
from theodb_bench.errors import UnsupportedCapabilityError
from theodb_bench.registry import ADAPTERS


def _docs() -> list[Document]:
    """Both legs of the retrieval surface: text and a vector, from one corpus."""
    import numpy as np

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "a lazy afternoon with a brown coffee",
        "quick sorting of very large arrays",
    ]
    rng = np.random.default_rng(20260817)
    return [
        Document(id=i, text=text, vector=rng.random(8, dtype=np.float32))
        for i, text in enumerate(texts)
    ]


DOCS = _docs()


class _PillarStub:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.copied_rows: list[tuple[Any, ...]] = []
        self.rows: list[tuple[Any, ...]] = [(0, 1.5), (2, 0.9)]

    def execute(self, sql: str, parameters: tuple[object, ...] | None = None) -> None:
        self.executed.append(sql)

    def fetch_one(
        self, sql: str, parameters: tuple[object, ...] | None = None
    ) -> tuple[object, ...] | None:
        # Recorded too: a statement sent through fetch_one is still a statement,
        # and a stub that only watched `execute` would miss half the surface.
        self.executed.append(sql)
        if "count(*)" in sql:
            return (len(DOCS),)
        if "bm25_build" in sql:
            return (1,)
        if "write_parquet" in sql:
            return (len(DOCS),)
        if "graph_expand_card" in sql:
            return (7,)
        if "pg_relation_size" in sql:
            return (4096,)
        return None

    def fetch_all(
        self, sql: str, parameters: tuple[object, ...] | None = None
    ) -> list[tuple[object, ...]]:
        self.executed.append(sql)
        return list(self.rows)

    def cursor(self) -> Any:
        return _PillarCursor(self)


class _PillarCursor:
    def __init__(self, server: _PillarStub) -> None:
        self._server = server

    def __enter__(self) -> _PillarCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def copy(self, sql: str) -> _PillarCopy:
        self._server.executed.append(sql)
        return _PillarCopy(self._server)

    def executemany(self, sql: str, batch: list[tuple[Any, ...]]) -> None:
        raise AssertionError("document load fell back to executemany")


class _PillarCopy:
    def __init__(self, server: _PillarStub) -> None:
        self._server = server

    def __enter__(self) -> _PillarCopy:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def write_row(self, row: tuple[Any, ...]) -> None:
        self._server.copied_rows.append(row)


def _wire(server: _PillarStub) -> TheoDBAdapter:
    adapter = TheoDBAdapter()
    adapter._execute = server.execute  # type: ignore[method-assign]
    adapter._fetch_one = server.fetch_one  # type: ignore[method-assign]
    adapter._fetch_all = server.fetch_all  # type: ignore[method-assign]
    adapter._cursor = server.cursor  # type: ignore[method-assign]
    return adapter


# ------------------------------------------------------------------- lexical


def test_theodb_declares_lexical_and_reaches_bm25() -> None:
    server = _PillarStub()
    adapter = _wire(server)
    spec = DocumentTableSpec(table="bench_docs", dimension=8)

    adapter.load_documents(spec, DOCS)
    result = adapter.execute_lexical(LexicalQuery(table="bench_docs", text="lazy brown", n=5))

    statements = " | ".join(server.executed)
    assert "bm25_build" in statements
    assert result.ids == (0, 2)
    assert result.scores == (1.5, 0.9)


def test_the_lexical_index_is_built_before_it_is_searched() -> None:
    """`bm25_search` on an index that was never built is the B-041 defect: it
    returned zero rows, indistinguishable from nothing matching.

    A mensagem mudou de "never built in this session" para "does not exist in the database"
    quando o B-043 mediu que a guarda perguntava a coisa errada — ver
    `test_the_guard_asks_the_database_not_the_instance`. A PROPRIEDADE e a mesma: buscar num
    indice que nao existe e recusado.
    """
    server = _PillarStub()
    adapter = _wire(server)

    with pytest.raises(UnsupportedCapabilityError, match="does not exist in the database"):
        adapter.execute_lexical(LexicalQuery(table="bench_docs", text="lazy", n=5))


# ------------------------------------------------------------------- parquet


def test_theodb_reaches_the_parquet_path() -> None:
    server = _PillarStub()
    adapter = _wire(server)
    table = AnalyticalTable(
        name="bench_analytical_parquet",
        columns=("id", "amount", "category", "quantity"),
        path="parquet",
    )

    adapter.load_analytical(table, [(0, 1.5, "a", 2)])

    statements = " | ".join(server.executed)
    assert "write_parquet" in statements


def test_a_parquet_query_reads_through_read_parquet() -> None:
    server = _PillarStub()
    adapter = _wire(server)
    table = AnalyticalTable(
        name="bench_analytical_parquet",
        columns=("id", "amount", "category", "quantity"),
        path="parquet",
    )

    sql = adapter._analytical_query_sql(table, _query("total_rows"))

    assert "read_parquet" in sql


def _query(query_id: str) -> Any:
    from theodb_bench.adapters.base import AnalyticalQuery

    return AnalyticalQuery(id=query_id, description="")


# ------------------------------------------- what stays undeclared, and why


@pytest.mark.parametrize("capability", ["rerank", "vectorizer", "ai_sql"])
def test_model_dependent_capabilities_stay_undeclared(capability: str) -> None:
    """Each of these reaches an external model. Without an endpoint there is
    nothing to measure, and a stub would put a number where an absence belongs."""
    for name, entry in ADAPTERS.items():
        if name == "fake":
            continue
        assert capability not in entry.factory().capabilities(), name


def test_every_declared_capability_is_still_in_the_vocabulary() -> None:
    for name, entry in ADAPTERS.items():
        if name == "fake":
            continue
        unknown = set(entry.factory().capabilities()) - set(CAPABILITIES)
        assert not unknown, f"{name}: {sorted(unknown)}"


# --------------------------------------------------------------------- graph
#
# Read from the running server:
#   theodb.graph_build(edge_rel text, src_col text, dst_col text) -> bigint
#   theodb.graph_expand(edge_rel text, seeds bigint[], max_hops int) -> SETOF bigint
#   theodb.graph_expand_card(edge_rel, seeds, max_hops) -> bigint


def test_theodb_builds_a_persisted_csr_before_traversing() -> None:
    server = _PillarStub()
    server.rows = [(1,), (2,), (5,)]
    adapter = _wire(server)
    # `directed=False` explicito: o CSR da extensao e nao-dirigido, e o adapter agora recusa
    # `directed=True` em vez de aceitar um pedido que nao pode honrar.
    spec = GraphSpec(name="bench_edges", directed=False)

    adapter.load_graph(spec, [(0, 1), (1, 2), (2, 5)], vertex_count=6)

    statements = " | ".join(server.executed)
    assert "graph_build" in statements


def test_traversing_a_graph_that_was_never_built_is_refused() -> None:
    """`graph_expand` over a relation with no persisted CSR would answer with an
    empty set, which reads as a vertex having no neighbours.

    A mensagem passou a falar do SERVIDOR ("no CSR exists ... on this server") e nao da sessao,
    porque a guarda passou a perguntar ao catalogo em vez de consultar um conjunto por instancia.
    O texto antigo afirmava sobre o servidor a partir da memoria do objeto.
    """
    adapter = _wire(_PillarStub())

    with pytest.raises(UnsupportedCapabilityError, match="no CSR exists"):
        adapter.traverse(TraversalQuery(graph="bench_edges", source=0, hops=2))


def test_a_traversal_reports_the_work_done_not_only_the_answer() -> None:
    """`edges_visited` is the cost. A traversal returning few vertices after
    walking many edges is expensive, and the answer size alone would hide it."""
    server = _PillarStub()
    server.rows = [(1,), (2,), (5,)]
    adapter = _wire(server)
    # `directed=False` explicito: o CSR da extensao e nao-dirigido, e o adapter agora recusa
    # `directed=True` em vez de aceitar um pedido que nao pode honrar.
    spec = GraphSpec(name="bench_edges", directed=False)
    adapter.load_graph(spec, [(0, 1), (1, 2), (2, 5)], vertex_count=6)

    result = adapter.traverse(TraversalQuery(graph="bench_edges", source=0, hops=2))

    assert result.vertices == (1, 2, 5)
    assert result.edges_visited > 0


# ------------------------------------------------------- hybrid and quantized


def test_theodb_fuses_both_legs_through_its_own_rrf() -> None:
    """`ai.hybrid_search_rrf` fuses inside the engine. The benchmark also fuses
    the legs itself, so the engine's fusion is checked rather than trusted."""
    server = _PillarStub()
    adapter = _wire(server)
    spec = DocumentTableSpec(table="pillar_docs", dimension=8)
    adapter.load_documents(spec, DOCS)

    result = adapter.execute_hybrid(
        HybridQuery(table="pillar_docs", text="lazy dog", vector=DOCS[0].vector, n=3)
    )

    statements = " | ".join(server.executed)
    assert "hybrid_search_rrf" in statements
    assert result.ids == (0, 2)


def test_hybrid_needs_both_legs_loaded() -> None:
    """Fusing over a corpus that only has text would fuse one leg with nothing."""
    adapter = _wire(_PillarStub())

    with pytest.raises(UnsupportedCapabilityError, match="does not exist in the database"):
        adapter.execute_hybrid(
            HybridQuery(table="pillar_docs", text="lazy", vector=DOCS[0].vector, n=3)
        )


def test_quantized_indexes_are_declared_because_the_suites_build_them() -> None:
    """`pq_subspaces`, `sbq_bits` and `rabitq_bits` are real reloptions, and
    `vector/sift/pg-scann` builds with `pq_subspaces=64`."""
    assert TheoDBAdapter().supports("vector_quantized")


# ---------------------------------------------------------- B-043: a guarda perguntava a instancia


class _ServidorComIndice(_PillarStub):
    """Um servidor que TEM o indice — o que o catalogo responderia depois de um `bm25_build`."""

    def fetch_one(
        self, sql: str, parameters: tuple[object, ...] | None = None
    ) -> tuple[object, ...] | None:
        if "lexical_index_meta" in sql:
            self.executed.append(sql)
            return (1,)
        return super().fetch_one(sql, parameters)


def test_the_guard_asks_the_database_not_the_instance() -> None:
    """O defeito que impedia QUALQUER curva de concorrencia.

    MEDIDO em 2026-08-21: a guarda consultava `self._lexical_built`, um conjunto POR INSTANCIA.
    Sob populacao de clientes, cada cliente novo nasce com ele vazio e **toda** consulta era
    recusada — 300 erros e zero sucessos ja a partir de dois clientes.

    A mensagem dizia "never built in this SESSION", mas o indice vive no BANCO. A guarda afirmava
    algo sobre a memoria do adapter e reportava como fato sobre o servidor.

    Este adapter e NOVO — nunca chamou `build_lexical_index` — e mesmo assim tem de conseguir
    buscar, porque o indice existe no banco.
    """
    adapter = _wire(_ServidorComIndice())
    assert adapter._lexical_built == set(), (
        "o teste so mede o que se propoe se a instancia for nova"
    )

    resultado = adapter.execute_lexical(LexicalQuery(table="bench_docs", text="lazy", n=5))
    assert resultado.ids == (0, 2)


def test_the_positive_answer_is_remembered_and_the_negative_is_not() -> None:
    """Memorizar o positivo evita um round-trip por consulta no caminho que a corrida MEDE.

    O negativo nao pode ser memorizado: a corrida pode construir o indice depois, e um "nao existe"
    guardado faria toda consulta seguinte falhar contra um indice que ja existe.
    """
    servidor = _ServidorComIndice()
    adapter = _wire(servidor)

    adapter.execute_lexical(LexicalQuery(table="bench_docs", text="lazy", n=5))
    adapter.execute_lexical(LexicalQuery(table="bench_docs", text="dog", n=5))
    consultas_ao_catalogo = [s for s in servidor.executed if "lexical_index_meta" in s]
    assert len(consultas_ao_catalogo) == 1, "o positivo e perguntado UMA vez"

    vazio = _wire(_PillarStub())
    for _ in range(2):
        with pytest.raises(UnsupportedCapabilityError):
            vazio.execute_lexical(LexicalQuery(table="bench_docs", text="lazy", n=5))
    assert vazio._lexical_built == set(), (
        "um negativo memorizado quebraria o indice construido depois"
    )


def test_the_lexical_index_id_is_stable_across_processes() -> None:
    """O id chaveia um objeto PERSISTENTE do banco, e vinha do `hash()` embutido.

    MEDIDO em 2026-08-21: o `hash()` de string em Python e aleatorizado por processo (PEP 456) —
    tres execucoes, tres ids diferentes para a mesma tabela. As consequencias:

    - um indice construido numa corrida NAO e encontravel na seguinte;
    - um gerador de carga externo (o `pgbench` que o DoD do B-043 exige) nao tem como computar o
      mesmo id;
    - dentro de UM processo tudo funciona, que e por que ninguem viu.

    Este teste roda o calculo num SUBPROCESSO com semente de hash diferente. Compara-lo consigo
    mesmo no processo atual nao provaria nada — o `hash()` tambem e estavel dentro de um processo.
    """
    import subprocess
    import sys

    from theodb_bench.adapters.postgres import TheoDBAdapter

    aqui = TheoDBAdapter.lexical_index_id("bench_documents")
    programa = (
        "import sys; sys.path.insert(0, 'src');"
        "from theodb_bench.adapters.postgres import TheoDBAdapter;"
        "print(TheoDBAdapter.lexical_index_id('bench_documents'))"
    )
    vistos = set()
    for semente in ("0", "1", "12345"):
        saida = subprocess.run(
            [sys.executable, "-c", programa],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={"PYTHONHASHSEED": semente, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        vistos.add(int(saida.stdout.strip()))
    assert vistos == {aqui}, (
        f"o id mudou entre processos: {vistos} contra {aqui} — um indice construido numa corrida "
        f"nao seria encontravel na seguinte"
    )
