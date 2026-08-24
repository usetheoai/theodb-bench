"""Todo benchmark registrado aceita o que o runner passa — verificado por assinatura.

**O defeito que isto impede, custou um droplet em 2026-08-24.** O runner chama `benchmark.points(...)`
sem saber que tipo segura — o docstring do protocolo diz isso explicitamente: *"the runner has no way
to know which kind it holds, and asking would put the regime back in the caller"*. Eu acrescentei
`index_repetitions` ao lado do runner e ao benchmark VETORIAL, e mais nenhum.

O resultado não foi um erro legível: o smoke analítico morreu com **zero pontos medidos** e a validação
reportou `sut_alive: FAIL — system under test crashed or became unreachable`. O servidor estava vivo e
respondendo; a mensagem apontava para o lugar errado, e a suíte local passava inteira.

Este teste compara a assinatura de CADA implementação com a do protocolo. É barato, roda em
milissegundos, e teria falhado no commit em que eu introduzi a divergência.
"""

from __future__ import annotations

import inspect

import pytest
from theodb_bench.bench.protocol import Benchmark
from theodb_bench.registry import BENCHMARKS


def _implementacoes():
    """Uma classe de BENCHMARK por tipo registrado, sem duplicar.

    A classe vem da anotacao de retorno de `workload.build` — o workload NAO e o benchmark, e
    inspecionar o workload faz cada caso PULAR por nao ter `points`. A primeira versao deste arquivo
    fez exatamente isso e passou com quatro `skip`, que e o defeito que ele existe para impedir.
    """
    vistas: dict[str, type] = {}
    for entry in BENCHMARKS.values():
        cls = inspect.signature(type(entry.workload).build).return_annotation
        if isinstance(cls, str):  # anotacao adiada
            import theodb_bench.bench as pacote
            for mod in ("vector", "graph", "analytical", "retrieval"):
                alvo = getattr(getattr(pacote, mod, None), cls, None)
                if alvo is not None:
                    cls = alvo
                    break
        if not isinstance(cls, type):
            continue
        vistas.setdefault(f"{cls.__module__}.{cls.__name__}", cls)
    return vistas


def _params(fn) -> set[str]:
    return {
        n
        for n, p in inspect.signature(fn).parameters.items()
        if n != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }


@pytest.mark.parametrize("nome", sorted(_implementacoes()))
def test_points_aceita_tudo_que_o_protocolo_declara(nome: str) -> None:
    cls = _implementacoes()[nome]
    assert hasattr(cls, "points"), f"{nome} nao implementa points — o runner a chamaria mesmo assim"
    do_protocolo = _params(Benchmark.points)
    da_classe = _params(cls.points)
    faltando = do_protocolo - da_classe
    assert not faltando, (
        f"{nome}.points nao aceita {sorted(faltando)}, que o protocolo declara. O runner chama "
        f"todas as implementacoes igual e nao sabe qual segura — a divergencia vira TypeError em "
        f"tempo de corrida, com mensagem que nao aponta para a causa."
    )


def test_o_protocolo_declara_index_repetitions() -> None:
    """Guarda contra o teste acima passar trivialmente se alguem remover o parametro do protocolo."""
    assert "index_repetitions" in _params(Benchmark.points)


def test_ha_mais_de_uma_implementacao_conferida() -> None:
    """Com uma implementacao so, o teste parametrizado nao provaria nada sobre uniformidade."""
    assert len(_implementacoes()) >= 3, sorted(_implementacoes())
