"""A sonda de contencao tem de nomear uma consulta que o adapter REAL conhece.

MEDIDO em 2026-08-22, na primeira corrida do executor de contencao contra um servidor de verdade:
os dois regimes falharam com `0/200` operacoes de leitura, e o arnes recusou reportar — corretamente,
porque "uma razao contra `None` se le como 'sem contencao' quando significa 'nada rodou'".

A causa: `_contention_probe` inventava `query.id = f"contention-{index}"`, e o `PostgresAdapter`
resolve o SQL por `ANALYTICAL_SQL[query.id]` — onde esse id nunca esteve. Toda leitura levantava
`unknown analytical query`.

Os testes do executor passavam porque usavam o adapter `fake`, que NAO consulta `ANALYTICAL_SQL` e
aceita qualquer id. E o caso classico de suite verde sobre caminho que nao existe: o teste exercitava
um substituto sem a restricao que quebra.
"""

from __future__ import annotations

import pytest
from theodb_bench.adapters.base import AnalyticalQuery, AnalyticalTable
from theodb_bench.adapters.postgres import PostgresAdapter
from theodb_bench.cli import _contention_probe
from theodb_bench.errors import BenchError


def _tabela() -> AnalyticalTable:
    return AnalyticalTable(name="bench_contention", columns=("id", "value"), path="columnar")


def test_a_sonda_nomeia_consulta_que_o_adapter_real_conhece() -> None:
    """O teste que faltava: resolver o SQL da sonda com o adapter REAL, nao com o `fake`."""
    sql = PostgresAdapter()._analytical_query_sql(_tabela(), _contention_probe(0))
    assert "bench_contention" in sql, f"o SQL nomeia a tabela: {sql}"


def test_ids_diferentes_continuam_resolvendo() -> None:
    """O `index` varia por operacao. Se ele entrar no id da consulta a resolucao quebra — que foi
    exatamente o defeito."""
    adapter = PostgresAdapter()
    for i in (0, 1, 199):
        adapter._analytical_query_sql(_tabela(), _contention_probe(i))


def test_a_sonda_so_usa_colunas_que_a_tabela_de_contencao_tem() -> None:
    """A tabela de contencao e `(id, value)`. Uma sonda que agregue `amount` resolveria o template e
    falharia no servidor — erro mais tarde e mais caro de diagnosticar."""
    sql = PostgresAdapter()._analytical_query_sql(_tabela(), _contention_probe(0))
    assert "amount" not in sql and "category" not in sql, f"coluna inexistente na sonda: {sql}"


def test_uma_consulta_desconhecida_ainda_e_recusada_com_erro_tipado() -> None:
    """A recusa tem de continuar existindo — ela e o que fez o defeito APARECER em vez de produzir
    um numero errado."""
    with pytest.raises(BenchError) as exc:
        PostgresAdapter()._analytical_query_sql(
            _tabela(), AnalyticalQuery(id="nao-existe", description="x")
        )
    assert "unknown analytical query" in str(exc.value)
