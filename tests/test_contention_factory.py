"""A fabrica de clientes da contencao tem de entregar cliente CONECTADO.

MEDIDO em 2026-08-22, na terceira corrida contra um servidor real: a tabela existia (125 MiB, carga
verificada com ON_ERROR_STOP), a sonda resolvia, as duas operacoes funcionavam quando chamadas a mao —
e mesmo assim os dois regimes fecharam com 0/200.

A causa: `measure_contention` chama `make_reader()`/`make_writer()` e usa o resultado direto; ele NAO
inicia o cliente. E a fabrica do CLI fazia apenas `.build()`, que constroi o adapter sem conectar.
Toda operacao levantava, e o contador registrava 0 sucessos sem que o erro aparecesse.

O teste local que "funcionava" chamava prepare/start/wait_ready explicitamente — ou seja, exercitava
um caminho que o codigo de producao nao percorre. E a mesma classe do adapter `fake`: um substituto
sem a restricao que quebra.
"""

from __future__ import annotations

from theodb_bench.cli import _contention_client_factory


class _AdapterEspiao:
    """Registra a ordem das chamadas de ciclo de vida."""

    def __init__(self) -> None:
        self.chamadas: list[str] = []

    def prepare(self) -> None:
        self.chamadas.append("prepare")

    def start(self) -> None:
        self.chamadas.append("start")

    def wait_ready(self) -> None:
        self.chamadas.append("wait_ready")


def test_a_fabrica_entrega_cliente_conectado() -> None:
    espiao = _AdapterEspiao()
    fabrica = _contention_client_factory(lambda: espiao)

    cliente = fabrica()

    assert cliente is espiao
    assert espiao.chamadas == ["prepare", "start", "wait_ready"], (
        "um cliente sem `start` nao tem conexao, e toda operacao falha em silencio: "
        f"{espiao.chamadas}"
    )


def test_cada_chamada_entrega_um_cliente_novo() -> None:
    """Cada leitor e cada escritor precisa da PROPRIA conexao — a contencao que se mede e entre
    sessoes, e clientes compartilhando conexao mediriam serializacao do cliente, nao do servidor."""
    criados: list[_AdapterEspiao] = []

    def construir() -> _AdapterEspiao:
        a = _AdapterEspiao()
        criados.append(a)
        return a

    fabrica = _contention_client_factory(construir)
    fabrica()
    fabrica()

    assert len(criados) == 2, "cada chamada constroi um cliente novo"
    assert all(a.chamadas == ["prepare", "start", "wait_ready"] for a in criados)
