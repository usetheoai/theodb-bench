"""B-043 — o arnes nao sabia dirigir N clientes no retrieval, entao a hipotese nao era testavel.

Medido em 2026-08-13: o QPS lexical satura em ~20 clientes numa maquina de 16 vCPU — de 20 a 80 o
throughput nao cresce 1% e a p99 cresce 4x. **A causa nao esta medida**, e uma das tres candidatas e
o proprio cliente Python do arnes.

Sem o arnes saber produzir a curva, essa hipotese nao pode nem ser posta ao lado de um gerador
externo. O DoD do item exige os DOIS: se ambos saturarem no mesmo ponto o cliente esta absolvido; se
o externo continuar subindo, o teto e nosso.
"""

from __future__ import annotations

from theodb_bench.adapters.base import AnalyticalQuery  # noqa: F401  (mantém o import-linter feliz)
from theodb_bench.bench.retrieval import RetrievalWorkload


def _workload(sweep: tuple[int, ...]) -> RetrievalWorkload:
    return RetrievalWorkload(
        corpus_size=40, query_count=8, k=3, n=5, pipelines=("lexical",), client_sweep=sweep
    )


def test_without_a_sweep_the_workload_declares_one_client() -> None:
    """A propriedade que as suites existentes protegem."""
    payload = _workload(()).benchmark_payload()
    assert payload["workload"]["clients"] == 1
    assert "clients" not in payload["parameters"], (
        "declarar `clients` como parametro VARRIDO sugeriria varredura onde ha um ponto so"
    )


def test_a_sweep_declares_its_widest_point_and_the_whole_curve() -> None:
    """`workload.clients` e o pico — e o que descreve o regime; `parameters` e a curva."""
    payload = _workload((1, 5, 20, 80)).benchmark_payload()
    assert payload["workload"]["clients"] == 80
    assert payload["parameters"]["clients"] == [1, 5, 20, 80]


def test_one_client_reuses_the_open_connection() -> None:
    """Abrir uma segunda conexao para fazer o trabalho que a primeira ja fazia mudaria a medida."""
    from theodb_bench.load import client_pool

    abrir, fechar = client_pool("o-adapter", None, 1)
    assert abrir() == "o-adapter"
    assert fechar is None


def test_many_clients_without_a_way_to_open_connections_is_refused() -> None:
    """Serializar N clientes numa conexao mede a TRAVA DO CLIENTE e reporta como sendo o banco.

    Recusar e a unica leitura honesta: a corrida produziria uma curva plana que parece saturacao do
    servidor e e do arnes — exatamente a hipotese que o B-043 existe para separar.
    """
    import pytest
    from theodb_bench.errors import ConfigError
    from theodb_bench.load import client_pool

    with pytest.raises(ConfigError, match="connection per client"):
        client_pool("o-adapter", None, 8)


def test_the_vector_family_and_the_retrieval_family_share_one_implementation() -> None:
    """Duas copias da regra divergiriam no ponto que mais importa.

    O `VectorBenchmark._client_pool` passou a delegar ao `load.client_pool` quando o retrieval
    precisou do mesmo. Se alguem reintroduzir uma segunda copia, este teste nao a vera — mas o
    delegacao esta afirmada aqui para que a intencao fique registrada onde se le.
    """
    import inspect

    from theodb_bench.bench.vector import VectorBenchmark

    fonte = inspect.getsource(VectorBenchmark._client_pool)
    assert "client_pool(" in fonte, "o vetorial tem de DELEGAR, nao reimplementar"


def test_the_operation_count_scales_with_the_client_population() -> None:
    """O defeito que produziu uma conclusao publicada ERRADA.

    A primeira versao emitia um total FIXO de operacoes, independente do numero de clientes.
    Medido, back-to-back no mesmo processo:

        clientes | total FIXO em 300 | 300 POR cliente
               5 |             598,6 |           646,1
              20 |             570,2 |           801,8
              80 |         **277,7** |       **827,0**

    A 80 clientes, 300 operacoes sao **3,75 por cliente** — a abertura de conexao domina a janela
    medida, e a curva COLAPSA. O colapso e do desenho da medicao, nao do sistema.

    Isso me levou a publicar, no B-043, que "o teto de vazao e o cliente do arnes", com um numero
    de 6,5x que era artefato do mesmo defeito. Com a contagem escalando, a curva sobe e satura.
    """
    w = _workload((1, 5, 20))
    # 1 pipeline x 1 repeticao x (1+5+20) clientes x 8 consultas
    assert w.expected_operations(measured_points=3, repetitions=1) == 26 * w.query_count


def test_without_a_sweep_the_count_is_unchanged() -> None:
    """A propriedade que as suites de um cliente so protegem."""
    w = _workload(())
    assert w.expected_operations(measured_points=1, repetitions=3) == 3 * w.query_count
