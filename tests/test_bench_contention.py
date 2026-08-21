"""B-066 — a contencao escrita x scan nao era medivel.

Medido em 2026-08-17 e RE-MEDIDO em 2026-08-21, porque a premissa do item envelheceu: hoje existe
motor concorrente (`src/load.py`, `ThreadPoolExecutor`) e o `bench/vector.py` o usa. O que NAO
existe
e carga MISTA — um escritor concorrente com leitores. O `bench/analytical.py` continua sequencial.

Por que importa: a avaliacao independente do AlloyDB mediu uma INVERSAO — ligar o colunar PIOROU a
contencao a SF100 (29% contra 16% do row store), contra empate a SF10. E o unico numero do artigo em
que o colunar do concorrente sai pior, e portanto o mais interessante de responder — e o que nao
tinhamos instrumento para medir.

A regra que estes testes travam vem do [[B-060]] e do [[B-063]]: comparar contra linha de base de
OUTRA corrida e a classe de erro que os dois documentam. Aqui ela e impossivel por construcao.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from theodb_bench.bench.contention import (
    ContentionSpec,
    Regime,
    SideOutcome,
    measure_contention,
)
from theodb_bench.errors import MeasurementError
from theodb_bench.load import LoadModel


class _FakeClient:
    def __init__(self) -> None:
        self.ops = 0


def _issue(delay: float) -> Callable[[_FakeClient, int], None]:
    def issue(client: _FakeClient, index: int) -> None:
        client.ops += 1
        # Trabalho deterministico e minusculo: o teste afirma a ESTRUTURA do resultado, nao
        # latencias reais — cravar numeros de relogio num teste unitario seria medir a maquina.
        for _ in range(int(delay)):
            pass

    return issue


def _spec(
    *,
    read_ops: int = 8,
    write_ops: int = 4,
    regime: Regime = Regime.MEMORY_RESIDENT,
) -> ContentionSpec:
    return ContentionSpec(
        readers=LoadModel(clients=2),
        writers=LoadModel(clients=1),
        read_ops=read_ops,
        write_ops=write_ops,
        regime=regime,
    )


def test_both_sides_run_and_both_baselines_are_measured() -> None:
    fora = measure_contention(
        _spec(),
        make_reader=_FakeClient,
        issue_read=_issue(10),
        make_writer=_FakeClient,
        issue_write=_issue(10),
    )
    assert fora.read.isolated.successes == 8
    assert fora.read.concurrent.successes == 8
    assert fora.write.isolated.successes == 4
    assert fora.write.concurrent.successes == 4


def test_the_degradation_is_a_ratio_and_not_an_absolute() -> None:
    """Bullet 2: numero absoluto nao diz nada sem a linha de base, e uma razao carrega as duas."""
    fora = measure_contention(
        _spec(),
        make_reader=_FakeClient,
        issue_read=_issue(10),
        make_writer=_FakeClient,
        issue_write=_issue(10),
    )
    resumo = fora.as_dict()
    assert "p95_ratio" in resumo["read"]
    assert "p99_ratio" in resumo["read"]
    # E o absoluto continua disponivel ao lado — a razao esconde a escala, e as duas juntas nao.
    assert "isolated" in resumo["read"] and "concurrent" in resumo["read"]


def test_the_regime_is_recorded_and_not_inferred() -> None:
    """Bullet 3: os dois regimes tem de aparecer DISTINGUIDOS no artefato.

    O arnes nao tem como saber se o dado cabe no cache — isso depende do host, do `shared_buffers` e
    do tamanho da tabela. Quem monta a corrida sabe; entao o regime e DECLARADO, e o artefato o
    carrega. Inferi-lo seria adivinhar e publicar o palpite.
    """
    fora = measure_contention(
        _spec(regime=Regime.EXCEEDS_CACHE),
        make_reader=_FakeClient,
        issue_read=_issue(10),
        make_writer=_FakeClient,
        issue_write=_issue(10),
    )
    assert fora.as_dict()["regime"] == "exceeds-cache"


def test_a_side_outcome_without_its_isolated_baseline_is_refused() -> None:
    """Bullet 4, e ele e o que separa uma medicao de uma alegacao.

    Comparar contra linha de base de OUTRA corrida e a classe de erro que o B-060 e o B-063
    documentam: o host mudou, o cache mudou, a versao mudou. Aqui a linha de base e medida DENTRO da
    mesma chamada, entao a comparacao entre sessoes e impossivel por construcao — e este teste
    garante que ninguem monte o resultado a mao para contornar isso.
    """
    # `type: ignore` DELIBERADO: o tipo ja recusa isto estaticamente, e essa e a defesa primaria.
    # Este teste cobre o chamador DINAMICO — codigo sem anotacao, JSON desserializado —, que o tipo
    # nao alcanca. Passar `None` aqui e exatamente o caso que o guard existe para pegar.
    with pytest.raises(ValueError, match="linha de base"):
        SideOutcome(isolated=None, concurrent=None)  # type: ignore[arg-type]


def test_a_zero_operation_side_is_refused_rather_than_reported_as_no_contention() -> None:
    """Zero operacoes num lado produz percentis `None`, e uma razao contra `None` nao existe.

    Reportar isso como "sem contencao" seria transformar ausencia de medicao em resultado — o
    defeito exato que o `explain_scan` do produto recusa cometer."""
    with pytest.raises(ValueError, match="read_ops"):
        _spec(read_ops=0)


# ---------------------------------------------------------------------------
# A OUTRA PONTA: sem escrita por operacao no adapter, o arnes so mede fakes.
#
# `SystemAdapter` declara `insert_document`/`update_document_text` e o `PostgresAdapter` NAO os
# implementa — `postgres.py:1384` diz isso em texto. E a contencao que o B-066 quer medir e a do
# COLUNAR: INSERT na tabela analitica enquanto scans rodam, que e onde a avaliacao independente
# mediu a inversao.
#
# Um modulo de medicao sem chamador real e o defeito do `assert_index_used` ([[B-063]]) outra vez, e
# o portao de modulo orfao do [[B-071]] pegou este mesmo modulo antes destes testes existirem.
# ---------------------------------------------------------------------------


def test_the_postgres_family_can_append_one_analytical_row() -> None:
    """A escrita de primeiro plano que a contencao precisa, na tabela que o item se importa."""
    from theodb_bench.adapters.base import AnalyticalTable
    from theodb_bench.adapters.postgres import PostgresAdapter

    adapter = PostgresAdapter()
    tabela = AnalyticalTable(name="bench_analytical", columns=("a", "b"), path="columnar")
    sql, params = adapter.append_analytical_row_sql(tabela, (1, 2))
    assert sql.startswith("INSERT INTO"), f"a escrita tem de ser um INSERT, veio: {sql}"
    assert '"bench_analytical"' in sql, "o identificador tem de ser citado"
    assert sql.count("%s") == len(params) == 2, (
        "um placeholder por valor — a mesma regra que o `_query_parameters` fixou no B-063"
    )


def test_a_row_whose_width_does_not_match_the_columns_is_refused() -> None:
    """Falhar alto e cedo em vez de deixar o servidor recusar depois com mensagem pior."""
    from theodb_bench.adapters.base import AnalyticalTable
    from theodb_bench.adapters.postgres import PostgresAdapter

    tabela = AnalyticalTable(name="t", columns=("a", "b"), path="columnar")
    with pytest.raises(ValueError, match="colunas"):
        PostgresAdapter().append_analytical_row_sql(tabela, (1,))


# ---------------------------------------------------------------------------
# A FIACAO: um modulo de medicao sem caminho ate ele e codigo morto.
#
# O portao de modulo orfao do [[B-071]] pegou `bench.contention` assim que ele foi escrito, e o
# veredito estava certo — foi a segunda vez neste ciclo que um instrumento correto nao tinha
# chamador. Estes testes fixam o caminho.
# ---------------------------------------------------------------------------


def test_the_cli_exposes_a_contention_command() -> None:
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(
        ["contention", "--system", "fake", "--read-ops", "4", "--write-ops", "2"]
    )
    assert args.command == "contention"
    assert hasattr(args, "func"), "o subcomando precisa de um handler, senao ele nao roda"


def test_the_cli_default_regime_is_declared_and_not_guessed() -> None:
    """O regime nao tem default silencioso: quem monta a corrida declara, porque o arnes nao
    sabe."""
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(
        [
            "contention",
            "--system",
            "fake",
            "--read-ops",
            "4",
            "--write-ops",
            "2",
            "--regime",
            "exceeds-cache",
        ]
    )
    assert args.regime == "exceeds-cache"


def test_a_side_where_nothing_succeeded_is_refused_not_reported_as_null() -> None:
    """Percentil `null` sobre zero sucessos NAO e "sem contencao" — e "nada rodou".

    Encontrado rodando o comando de verdade contra o adapter fake, que nao implementa as operacoes:
    todas as tentativas erraram, `run_load` as registrou como erro (corretamente — um erro que some
    encolhe a amostra e transforma sistema quebrado em sistema rapido), e o meu relatorio devolveu
    `p95_ratio: null` sem dizer que nao havia medicao nenhuma por tras.

    E o defeito que este ciclo inteiro perseguiu, agora meu: um relatorio que nao distingue ausencia
    de resultado de resultado.
    """

    class _SempreFalha:
        pass

    def _explode(client: object, index: int) -> None:
        raise RuntimeError("esta operacao nao existe neste adapter")

    # `MeasurementError` e nao `ValueError`: o CLI trata `BenchError` e imprime o contexto em vez
    # de deixar um traceback fazer as vezes de diagnostico (`rules/error-handling.md` § 2).
    with pytest.raises(MeasurementError, match="nenhuma operação"):
        measure_contention(
            _spec(),
            make_reader=_SempreFalha,
            issue_read=_explode,
            make_writer=_SempreFalha,
            issue_write=_explode,
        )
