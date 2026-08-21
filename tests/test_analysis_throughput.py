"""B-049 — as diferencas de VELOCIDADE que publicamos nao tinham teste, e o pareado nao serve.

O [[B-045]] fechou a lacuna para QUALIDADE: a paridade lexical do `b047` tem p=0,477 sobre 6.980
consultas, com IC de [-0,0011, +0,0025]. Ele NAO fecha para velocidade, e a razao e estrutural: o
teste pareado precisa de valor POR CONSULTA, e QPS nao tem — e uma taxa agregada sobre a corrida
inteira.

As duas maiores diferencas que publicamos sao justamente de velocidade e seguem sem teste:
Elasticsearch faz 4,3x o nosso QPS no lexical, e o pgvector faz +16,3% a recall casado no vetorial.
A assimetria e indefensavel: exigimos rigor onde a diferenca e minuscula e o dispensamos onde ela e
de 4x.

O ERRO QUE ESTE MODULO EXISTE PARA NAO COMETER esta escrito no DoD: "nunca aplicar o pareado a taxas
agregadas". Duas corridas de QPS nao sao pares — sao amostras independentes, e trata-las como pares
inventa uma correlacao que nao existe e estreita o intervalo de confianca sem razao.
"""

from __future__ import annotations

import numpy as np
import pytest
from theodb_bench.analysis.throughput import (
    compare_throughput,
    precision_for_n,
    summarise_runs,
)


def test_a_single_run_cannot_produce_a_confidence_interval() -> None:
    """Uma corrida nao tem variancia, e um IC sobre uma amostra e uma ilusao de precisao.

    E o estado atual do `b047`: UMA corrida por configuracao. O modulo recusa em vez de devolver um
    intervalo de largura zero, que se leria como certeza absoluta.
    """
    with pytest.raises(Exception, match="pelo menos"):
        summarise_runs([1234.5])


def test_the_summary_carries_spread_and_not_just_the_centre() -> None:
    resumo = summarise_runs([100.0, 104.0, 96.0, 102.0, 98.0])
    assert resumo.n == 5
    assert 99.0 <= resumo.mean <= 101.0
    assert resumo.stdev > 0
    assert resumo.ci_low < resumo.mean < resumo.ci_high


def test_between_run_variance_is_reported_alongside_the_number() -> None:
    """Bullet 5: se a variancia entre corridas for grande, o numero publicado precisa dizer ISSO
    mais do que precisa de um `p`."""
    estavel = summarise_runs([100.0, 100.5, 99.5, 100.2, 99.8])
    instavel = summarise_runs([100.0, 140.0, 60.0, 130.0, 70.0])
    assert instavel.coefficient_of_variation > estavel.coefficient_of_variation
    assert instavel.as_dict()["coefficient_of_variation"] > 0.1


def test_two_clearly_different_systems_are_called_different() -> None:
    a = [100.0, 102.0, 98.0, 101.0, 99.0]
    b = [430.0, 435.0, 425.0, 432.0, 428.0]  # ~4,3x, a ordem do achado do b047
    r = compare_throughput(a, b, seed=7)
    assert r.significant
    assert r.ratio > 4.0
    assert r.ci_low > 1.0, "o IC da RAZAO nao pode cruzar 1 quando a diferenca e de 4x"


def test_two_overlapping_systems_are_not_called_different() -> None:
    """O contrapositivo importa tanto quanto: um teste que sempre diz 'diferente'
    nao distingue nada."""
    a = [100.0, 105.0, 95.0, 103.0, 97.0]
    b = [101.0, 104.0, 96.0, 102.0, 98.0]
    r = compare_throughput(a, b, seed=7)
    assert not r.significant


def test_the_test_used_is_named_in_the_result() -> None:
    """Quem le o artefato precisa saber QUAL teste produziu o numero — e por que
    ele, e nao o pareado."""
    r = compare_throughput([100.0, 102.0, 98.0], [200.0, 204.0, 196.0], seed=7)
    d = r.as_dict()
    assert "welch" in d["method"].lower() or "bootstrap" in d["method"].lower()
    assert d["paired"] is False


def test_sequences_of_different_length_are_accepted_because_samples_are_independent() -> None:
    """A diferenca estrutural com o pareado, num teste: amostras independentes nao precisam casar.

    O `compare_systems` pareado EXIGE mesmo comprimento e mesma ordem, porque o elemento i de cada
    lado e a mesma consulta. Aqui nao ha par nenhum — sao corridas separadas, e exigir simetria
    sugeriria uma correspondencia que nao existe.
    """
    r = compare_throughput([100.0, 102.0, 98.0], [200.0, 204.0, 196.0, 202.0, 198.0], seed=7)
    assert r.n_a == 3 and r.n_b == 5


def test_precision_for_n_says_what_each_extra_run_buys() -> None:
    """Bullet 3: o item tem de dizer qual N compra qual precisao, em vez de escolher por habito.

    Custo medido no proprio item: cada corrida do caso FTS levou ~7 min no droplet depois do dataset
    em cache. Saber que N=5 estreita o IC para X% e o que torna essa conta uma decisao.
    """
    largo = precision_for_n(cv=0.05, n=3)
    estreito = precision_for_n(cv=0.05, n=10)
    assert largo > estreito, "mais corridas tem de comprar um intervalo mais estreito"
    assert 0.0 < estreito < largo < 1.0


def test_precision_for_n_refuses_a_sample_too_small_to_have_a_t_value() -> None:
    with pytest.raises(Exception, match="pelo menos"):
        precision_for_n(cv=0.05, n=1)


# ---------------------------------------------------------------------------
# VALIDACAO contra referencia independente.
#
# Estatistica escrita a mao que ninguem validou e pior que nenhuma: ela produz numeros com a mesma
# aparencia de autoridade e sem a propriedade. O `_welch_p_value` e o `_t_critical` sao
# implementacoes proprias (para nao trazer SciPy por duas funcoes), entao a correcao deles precisa
# ser AFIRMADA contra algo externo.
#
# O SciPy nao e dependencia declarada deste projeto — quando ele nao esta, o teste PULA dizendo por
# que, em vez de passar em silencio e sugerir uma verificacao que nao houve.
# ---------------------------------------------------------------------------

# O `importorskip` fica DENTRO do teste que precisa dele, e nao no modulo. MEDIDO: com ele no topo,
# o venv deste projeto (sem SciPy) pulava o arquivo INTEIRO — "1 skipped", zero dos ~20 testes do
# B-049 executados, e o relatorio ainda parecia verde. Um portao que desliga o que nao dependia dele
# e a mesma classe de defeito que este item existe para nao repetir.
_MOTIVO_SEM_SCIPY = (
    "SciPy nao e dependencia declarada; sem ele a validacao externa nao roda e isto fica dito"
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([100.0, 102.0, 98.0, 101.0, 99.0], [430.0, 435.0, 425.0, 432.0, 428.0]),
        ([100.0, 105.0, 95.0, 103.0, 97.0], [101.0, 104.0, 96.0, 102.0, 98.0]),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0]),
        ([10.0, 12.0, 11.0, 13.0], [10.5, 11.5, 12.5, 9.5, 13.5, 11.0]),
    ],
)
def test_the_welch_p_value_matches_scipy(a: list[float], b: list[float]) -> None:
    scipy_stats = pytest.importorskip("scipy.stats", reason=_MOTIVO_SEM_SCIPY)

    from theodb_bench.analysis.throughput import _welch_p_value

    meu = _welch_p_value(np.array(a), np.array(b))
    referencia = float(scipy_stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert meu == pytest.approx(referencia, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize(
    ("df", "esperado"),
    [(1, 12.706), (2, 4.303), (5, 2.571), (10, 2.228), (30, 2.042), (100, 1.984)],
)
def test_the_t_critical_matches_the_published_table(df: int, esperado: float) -> None:
    """Valores tabelados classicos de t critico bicaudal a 95%."""
    from theodb_bench.analysis.throughput import _t_critical

    assert _t_critical(df, 0.05) == pytest.approx(esperado, abs=0.001)


# ---------------------------------------------------------------------------
# A FIACAO: o veredito de vazao ao lado do de qualidade, e dizendo que sao diferentes.
# ---------------------------------------------------------------------------


def test_the_throughput_verdict_says_which_test_produced_it() -> None:
    """Quem le tem de saber que este NAO e o pareado, senao a distincao inteira se perde."""
    from theodb_bench.compare import render_throughput_verdict

    texto = render_throughput_verdict(
        "theodb",
        [100.0, 102.0, 98.0, 101.0, 99.0],
        "elasticsearch",
        [430.0, 435.0, 425.0, 432.0, 428.0],
    )
    assert "4.3" in texto or "4,3" in texto
    assert "Welch" in texto
    assert "unpaired" in texto.lower() or "nao pareado" in texto.lower()


def test_a_single_run_per_side_is_refused_with_the_reason() -> None:
    """O estado atual do `b047` e UMA corrida por configuracao — e ele nao vira veredito."""
    from theodb_bench.compare import render_throughput_verdict

    texto = render_throughput_verdict("a", [100.0], "b", [430.0])
    assert "not comparable" in texto.lower() or "nao comparavel" in texto.lower()
    assert "1" in texto


def test_a_high_variance_result_warns_before_it_concludes() -> None:
    """Bullet 5: variancia grande entre corridas precisa ser dita ANTES do `p`."""
    from theodb_bench.compare import render_throughput_verdict

    texto = render_throughput_verdict(
        "a",
        [100.0, 160.0, 40.0, 150.0, 50.0],
        "b",
        [430.0, 700.0, 160.0, 650.0, 200.0],
    )
    assert "variance" in texto.lower() or "variância" in texto.lower()


def test_the_cli_exposes_a_throughput_command_taking_n_runs_per_side() -> None:
    """Bullet 1: N corridas por configuracao, com N DECLARADO — nao um numero solto."""
    from theodb_bench.cli import build_parser

    args = build_parser().parse_args(
        [
            "throughput",
            "--a",
            "theodb",
            "--a-runs",
            "100",
            "102",
            "98",
            "--b",
            "elasticsearch",
            "--b-runs",
            "430",
            "435",
            "425",
        ]
    )
    assert args.command == "throughput"
    assert len(args.a_runs) == 3 and len(args.b_runs) == 3
