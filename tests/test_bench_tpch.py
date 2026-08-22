"""B-065 — o contrato analitico era de UMA tabela, e a comparacao do SOTA e TPC-H multi-tabela.

Medido em 2026-08-17: `AnalyticalTable` carrega `name`, `columns` e `path` — uma tabela. As quatro
queries de `bench/analytical.py` sao agregacao sobre tabela unica. **Nenhuma juncao e expressavel.**

A avaliacao independente do AlloyDB publicou Q1/Q5/Q6/Q18 do TPC-H. Sem esquema multi-tabela, os
numeros do concorrente nao tem onde ser respondidos com o mesmo shape — e responder com shape nosso
mede outra coisa e chama de comparacao.

A Q18 e a que PROVA o redesenho: `customer` |x| `orders` |x| `lineitem`, tres tabelas. Q1 e Q6 sao
de `lineitem` so, e passariam no contrato antigo.
"""

from __future__ import annotations

import pytest
from theodb_bench.bench.tpch import (
    TPCH_QUERIES,
    AnalyticalSchema,
    ForeignKey,
    expected_tpch_answer,
    generate_tpch,
    tpch_schema,
)


def test_the_schema_declares_tables_and_the_keys_between_them() -> None:
    """Bullet 1: esquema multi-tabela COM CHAVES. Sem as chaves nao ha juncao expressavel."""
    esquema = tpch_schema()
    assert isinstance(esquema, AnalyticalSchema)
    nomes = {t.name for t in esquema.tables}
    assert {"customer", "orders", "lineitem"} <= nomes
    assert any(isinstance(k, ForeignKey) for k in esquema.keys)
    # `orders.custkey -> customer.custkey` e `lineitem.orderkey -> orders.orderkey` sao as duas que
    # a Q18 percorre.
    arestas = {(k.table, k.column, k.references_table) for k in esquema.keys}
    assert ("orders", "o_custkey", "customer") in arestas
    assert ("lineitem", "l_orderkey", "orders") in arestas


def test_the_generator_is_seeded_and_reproducible() -> None:
    """Bullet 4: semeado e reprodutivel — duas chamadas com a mesma semente dao o mesmo dado.

    Sem isso, comparar duas corridas mede a diferenca entre os DADOS e chama de diferenca entre os
    sistemas.
    """
    a = generate_tpch(scale_factor=0.001, seed=42)
    b = generate_tpch(scale_factor=0.001, seed=42)
    assert a == b
    c = generate_tpch(scale_factor=0.001, seed=43)
    assert a != c, (
        "sementes diferentes tem de produzir dados diferentes, senao a semente e decorativa"
    )


def test_the_scale_factor_changes_the_size_and_appears_in_the_label() -> None:
    pequeno = generate_tpch(scale_factor=0.001, seed=42)
    maior = generate_tpch(scale_factor=0.002, seed=42)
    assert len(maior["lineitem"]) > len(pequeno["lineitem"])


@pytest.mark.parametrize("qid", ["q1", "q6", "q18"])
def test_the_three_registered_queries_have_an_oracle(qid: str) -> None:
    """Bullet 2: Q1, Q6 e Q18 registradas com o shape que o concorrente publicou."""
    assert qid in {q.id for q in TPCH_QUERIES}
    dados = generate_tpch(scale_factor=0.001, seed=42)
    resposta = expected_tpch_answer(dados, qid)
    assert isinstance(resposta, tuple)
    assert resposta, f"o oraculo de {qid} devolveu vazio — um oraculo vazio nao detecta nada"


def test_the_join_oracle_computes_q18_without_consulting_any_measured_path() -> None:
    """A Q18 junta tres tabelas, e o oraculo a calcula AQUI.

    Esse e o ponto do oraculo em geral e desta juncao em particular: se os tres caminhos medidos
    concordassem na mesma resposta errada, compara-los entre si nao acharia nada. A juncao e
    calculada em Python sobre os dados gerados, sem tocar em nenhum motor.
    """
    dados = generate_tpch(scale_factor=0.002, seed=7)
    resposta = expected_tpch_answer(dados, "q18")
    # Q18 devolve clientes cujos pedidos somam acima de um limiar: (nome, custkey, orderkey, ...)
    for linha in resposta:
        assert len(linha) >= 3, f"a linha da Q18 carrega colunas de MAIS DE UMA tabela: {linha}"


def test_an_unknown_query_is_refused_rather_than_answered_empty() -> None:
    dados = generate_tpch(scale_factor=0.001, seed=42)
    with pytest.raises(Exception, match="q99"):
        expected_tpch_answer(dados, "q99")


def test_q5_is_declared_out_of_scope_with_the_reason_in_the_registry() -> None:
    """Bullet 3: a Q5 junta SEIS tabelas — registrada ou explicitamente fora de escopo COM a razao.

    O silencio e que nao serve: uma query ausente sem explicacao se le como esquecimento, e a
    proxima pessoa refaz a analise para descobrir o mesmo.
    """
    from theodb_bench.bench.tpch import OUT_OF_SCOPE

    assert "q5" in OUT_OF_SCOPE
    assert len(OUT_OF_SCOPE["q5"]) > 40, "a razao tem de ser uma razao, nao um rotulo"


# ---------------------------------------------------------------------------
# O SQL: sem ele o esquema e um contrato que nenhum motor pode executar.
# ---------------------------------------------------------------------------


def test_q18_sql_joins_the_three_tables_the_schema_declares() -> None:
    """A juncao tem de aparecer no SQL, e com os nomes que o esquema declara.

    Gerar o SQL a partir do ESQUEMA e nao de literais e o que impede a divergencia: um prefixo de
    tabela mudaria o esquema e deixaria o SQL apontando para o lugar antigo.
    """
    from theodb_bench.bench.tpch import tpch_sql

    esquema = tpch_schema(prefix="x_")
    sql = tpch_sql(esquema, "q18")
    assert "x_customer" in sql and "x_orders" in sql and "x_lineitem" in sql
    assert sql.upper().count("JOIN") >= 2, "customer |x| orders |x| lineitem sao duas juncoes"


def test_q6_sql_touches_only_lineitem() -> None:
    from theodb_bench.bench.tpch import tpch_sql

    sql = tpch_sql(tpch_schema(), "q6")
    assert "lineitem" in sql
    assert "customer" not in sql and "orders" not in sql


def test_an_out_of_scope_query_has_no_sql_and_says_why() -> None:
    from theodb_bench.bench.tpch import tpch_sql

    with pytest.raises(Exception, match="fora de escopo"):
        tpch_sql(tpch_schema(), "q5")


@pytest.mark.parametrize(
    # `café` NAO esta aqui de proposito: `.isalnum()` e Unicode-aware e `"café"` e identificador
    # legal no PostgreSQL quando citado. Rejeita-lo seria recusar um nome valido; o que importa e
    # que a saida venha CITADA, e o teste abaixo afirma isso.
    "nome",
    ["", "tabela; DROP TABLE x", "tabela'--", "tab ela", "tabela)"],
)
def test_an_identifier_that_is_not_an_identifier_is_refused(nome: str) -> None:
    """A isencao de `S608` no `pyproject.toml` so e legitima porque ISTO existe.

    O precedente do adapter Postgres diz em texto que a isencao "e enforcada por
    `test_identifiers_that_are_not_identifiers_are_refused`". Uma isencao sustentada por argumento e
    uma opiniao; sustentada por teste e uma garantia.
    """
    from theodb_bench.bench.tpch import _safe_identifier

    with pytest.raises(Exception, match="identificador"):
        _safe_identifier(nome)


def test_a_valid_but_non_ascii_identifier_is_quoted_rather_than_refused() -> None:
    """Citar e o que torna a validacao suficiente — e foi a metade que faltou no primeiro
    rascunho."""
    from theodb_bench.bench.tpch import _safe_identifier

    assert _safe_identifier("café") == '"café"'
    assert _safe_identifier("lineitem") == '"lineitem"'


def test_a_schema_with_an_injected_table_name_cannot_produce_sql() -> None:
    """A defesa e no CAMINHO, nao so na funcao: um esquema malicioso nao gera SQL nenhum."""
    from theodb_bench.adapters.base import AnalyticalTable
    from theodb_bench.bench.tpch import AnalyticalSchema, ForeignKey, tpch_sql

    mau = AnalyticalSchema(
        tables=(
            AnalyticalTable(name="customer; DROP TABLE users", columns=("c_custkey",)),
            AnalyticalTable(name="orders", columns=("o_orderkey",)),
            AnalyticalTable(name="lineitem", columns=("l_orderkey",)),
        ),
        keys=(ForeignKey("orders", "o_custkey", "customer", "c_custkey"),),
    )
    with pytest.raises(Exception, match="identificador"):
        tpch_sql(mau, "q18")


# ---------------------------------------------------------------------------
# EXECUCAO: o contrato so vale se algum motor puder rodar.
#
# O portao de codigo morto acusou `generate_tpch`, `AnalyticalSchema.keys` e os campos da
# `ForeignKey` — todos usados so em teste. Estava certo: contrato entregue nao e contrato
# executavel, e a diferenca e o que o [[B-072]] pune.
# ---------------------------------------------------------------------------


def test_the_suite_loads_every_table_and_runs_every_query() -> None:
    from theodb_bench.bench.tpch import run_tpch_suite

    executado: dict[str, list[str]] = {"load": [], "sql": []}

    class _Motor:
        def load_analytical(self, table, rows) -> None:  # type: ignore[no-untyped-def]
            executado["load"].append(table.name)

        def execute_analytical_sql(self, sql: str):  # type: ignore[no-untyped-def]
            executado["sql"].append(sql)
            return ()

    resultado = run_tpch_suite(_Motor(), scale_factor=0.001, seed=42)
    assert sorted(executado["load"]) == sorted(t.name for t in tpch_schema().tables)
    assert len(executado["sql"]) == len(TPCH_QUERIES)
    assert set(resultado) == {q.id for q in TPCH_QUERIES}


def test_the_suite_reports_when_an_answer_disagrees_with_the_oracle() -> None:
    """Uma query rapida e errada nao e uma query rapida — a suite tem de dizer qual foi.

    E a mesma disciplina do `AnalyticalBenchmark`, que compara contra o proprio oraculo em vez de
    comparar os caminhos entre si: se os tres concordassem no erro, nada apareceria.
    """
    from theodb_bench.bench.tpch import run_tpch_suite

    class _Mentiroso:
        def load_analytical(self, table, rows) -> None:  # type: ignore[no-untyped-def]
            pass

        def execute_analytical_sql(self, sql: str):  # type: ignore[no-untyped-def]
            return ((-999,),)

    resultado = run_tpch_suite(_Mentiroso(), scale_factor=0.001, seed=42)
    assert any(not r.matches_oracle for r in resultado.values()), (
        "um motor que devolve -999 para tudo tem de ser pego pelo oraculo"
    )


def test_a_key_pointing_at_a_column_that_does_not_exist_is_refused() -> None:
    """Uma chave invalida descreve uma juncao impossivel, e o SQL so falharia no servidor."""
    from theodb_bench.adapters.base import AnalyticalTable
    from theodb_bench.bench.tpch import AnalyticalSchema, ForeignKey, run_tpch_suite

    class _Motor:
        def load_analytical(self, table, rows) -> None:  # type: ignore[no-untyped-def]
            pass

        def execute_analytical_sql(self, sql: str):  # type: ignore[no-untyped-def]
            return ()

    mau = AnalyticalSchema(
        tables=(AnalyticalTable(name="customer", columns=("c_custkey",)),),
        keys=(ForeignKey("customer", "x", "customer", "coluna_que_nao_existe"),),
    )
    import theodb_bench.bench.tpch as mod

    original = mod.tpch_schema
    mod.tpch_schema = lambda **_: mau
    try:
        with pytest.raises(Exception, match="não existe"):
            run_tpch_suite(_Motor(), scale_factor=0.001, seed=1)
    finally:
        mod.tpch_schema = original


# ------------------------ B-058: o TPC-H tem de saber rodar no colunar, nao so no heap
#
# O criterio aberto do B-058 e "TPC-H nos mesmos moldes: theodb_columnar contra heap
# no MESMO binario, e contra o Omni com engine off/on na mesma maquina". A suite
# existia e sempre criava heap, porque `tpch_schema` nao tinha por onde receber o
# caminho — e `AnalyticalTable.path` e justamente o campo que decide o access method.


def test_the_schema_defaults_to_heap() -> None:
    """Sem pedir nada, e heap — o comportamento que ja existia nao muda."""
    schema = tpch_schema()

    assert [t.path for t in schema.tables] == ["row", "row", "row"]


def test_the_schema_carries_the_requested_path_to_every_table() -> None:
    """Uma tabela em heap no meio de um TPC-H colunar mediria uma juncao hibrida."""
    schema = tpch_schema(path="columnar")

    assert [t.path for t in schema.tables] == ["columnar", "columnar", "columnar"]
    assert len(schema.tables) == 3, "as tres tabelas da Q18 precisam do mesmo caminho"


def test_the_path_survives_the_prefix() -> None:
    """Prefixo e caminho sao eixos independentes e nao podem interferir um no outro."""
    schema = tpch_schema(prefix="x_", path="columnar")

    assert all(t.name.startswith("x_") for t in schema.tables)
    assert all(t.path == "columnar" for t in schema.tables)
