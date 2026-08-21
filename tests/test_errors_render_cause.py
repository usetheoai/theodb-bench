"""A causa de um erro tem de aparecer na mensagem LEGIVEL, nao so no JSON.

MEDIDO em 2026-08-21 numa corrida de droplet: o bundle registrou
`could not connect to theodb [phase=bootstrap system=theodb]` e nada mais. A causa real,
`FATAL: role "root" does not exist`, estava em `cause` e no `as_dict()` — mas o `system.log`
formata `{exc}`, entao ninguem a via. Diagnosticar custou DUAS corridas de benchmark.
"""

from __future__ import annotations

from theodb_bench.errors import ErrorContext, Phase, SystemUnavailableError


def test_the_readable_message_carries_the_cause() -> None:
    erro = SystemUnavailableError(
        "could not connect to theodb",
        context=ErrorContext(phase=Phase.BOOTSTRAP, system="theodb"),
        cause=RuntimeError('FATAL:  role "root" does not exist'),
    )
    texto = str(erro)
    assert "could not connect to theodb" in texto
    assert "phase=bootstrap" in texto
    assert 'role "root" does not exist' in texto, (
        "sem a causa no texto, quem le o log ve apenas que a conexao falhou e nao POR QUE"
    )


def test_an_error_without_a_cause_is_unchanged() -> None:
    """Nao acrescentar ruido quando nao ha causa: o formato antigo continua valendo."""
    erro = SystemUnavailableError(
        "could not connect to theodb",
        context=ErrorContext(phase=Phase.BOOTSTRAP, system="theodb"),
    )
    assert str(erro) == "could not connect to theodb [phase=bootstrap system=theodb]"


def test_the_structured_payload_still_carries_it_too() -> None:
    """O JSON ja carregava a causa e continua carregando — o conserto ACRESCENTA, nao troca."""
    erro = SystemUnavailableError(
        "boom",
        context=ErrorContext(phase=Phase.BOOTSTRAP, system="theodb"),
        cause=ValueError("porque sim"),
    )
    assert "porque sim" in erro.as_dict()["cause"]
