"""Contenção escrita x scan: dois lados ao mesmo tempo, medidos contra a própria linha de base.

POR QUE ESTE MÓDULO EXISTE (B-066). A avaliação independente do AlloyDB mediu uma **inversão**:
ligar o colunar **piorou** a contenção a SF100 (29% contra 16% do row store), contra empate a SF10.
É o único número do artigo em que o colunar do concorrente sai pior, e portanto o mais interessante
de responder — e o arnês não tinha instrumento para medi-lo.

A premissa do item envelheceu e vale dizer: quando ele foi registrado, `grep -rn "thread"` em
`src/bench/` não devolvia nada. Hoje existe motor concorrente (`load.run_load`, com
`ThreadPoolExecutor`) e o `bench/vector.py` o usa. O que faltava — e é o que este módulo entrega —
é carga **mista**: um escritor rodando ao mesmo tempo que leitores.

CONSTRUÍDO SOBRE `run_load`, NÃO AO LADO DELE. Um segundo executor divergiria do primeiro, que é o
defeito que o `_query_parameters` do adapter Postgres pagou no mesmo ciclo. Aqui as duas cargas são
duas invocações do mesmo motor, em threads separadas.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from theodb_bench.errors import ErrorContext, MeasurementError, Phase
from theodb_bench.load import LoadModel, LoadResult, run_load, summarise_load


class Regime(str, Enum):
    """Se o dado cabe no cache do servidor, ou não.

    DECLARADO e nunca inferido. O arnês não tem como saber: depende do host, do `shared_buffers` e
    do tamanho da tabela. Quem monta a corrida sabe; inferir seria adivinhar e publicar o palpite.
    """

    MEMORY_RESIDENT = "memory-resident"
    EXCEEDS_CACHE = "exceeds-cache"


@dataclass(frozen=True)
class ContentionSpec:
    """Como as duas cargas são emitidas."""

    readers: LoadModel
    writers: LoadModel
    read_ops: int
    write_ops: int
    regime: Regime

    def __post_init__(self) -> None:
        # Zero operações num lado produz percentis `None`, e uma razão contra `None` não existe.
        # Reportar isso como "sem contenção" transformaria ausência de medição em resultado.
        if self.read_ops < 1:
            raise ValueError(f"read_ops precisa de pelo menos 1 operação, recebeu {self.read_ops}")
        if self.write_ops < 1:
            raise ValueError(
                f"write_ops precisa de pelo menos 1 operação, recebeu {self.write_ops}"
            )


@dataclass(frozen=True)
class SideOutcome:
    """Um lado da contenção: como ele se comporta sozinho e acompanhado."""

    isolated: LoadResult
    concurrent: LoadResult

    def __post_init__(self) -> None:
        # O TIPO ja recusa `None` estaticamente, e isso e mais forte que este guard — o mypy
        # reprova antes de rodar. O guard fica para o chamador dinamico, que o tipo nao alcanca.
        #
        # A regra do B-060 e do B-063, aplicada por construção: comparar contra linha de base de
        # OUTRA corrida é a classe de erro que os dois documentam — o host mudou, o cache mudou, a
        # versão mudou. Aqui a linha de base é medida dentro da mesma chamada; este guard existe
        # para que ninguém monte o resultado à mão e contorne isso.
        if self.isolated is None or self.concurrent is None:
            raise ValueError(
                "contenção sem linha de base isolada medida na MESMA sessão não é reportável: "
                "uma razão contra baseline de outra corrida mede a diferença entre as corridas, "
                "não a contenção"
            )

    def _ratio(self, chave: str) -> float | None:
        base = summarise_load(self.isolated)[chave]
        sob = summarise_load(self.concurrent)[chave]
        if base is None or sob is None or base == 0:
            return None
        return float(sob) / float(base)

    def as_dict(self) -> dict[str, Any]:
        return {
            "isolated": summarise_load(self.isolated),
            "concurrent": summarise_load(self.concurrent),
            # RAZÃO e não número absoluto (bullet 2 do B-066): "p95 de 40 ms" não diz nada sem a
            # linha de base, e "1,8x" carrega as duas. O absoluto fica ao lado porque a razão
            # esconde a escala, e uma degradação de 1,8x sobre 2 ms não é a mesma notícia que sobre
            # 200 ms.
            "p95_ratio": self._ratio("response_p95_ms"),
            "p99_ratio": self._ratio("response_p99_ms"),
        }


@dataclass(frozen=True)
class ContentionOutcome:
    read: SideOutcome
    write: SideOutcome
    regime: Regime

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "read": self.read.as_dict(),
            "write": self.write.as_dict(),
        }


def measure_contention(
    spec: ContentionSpec,
    *,
    make_reader: Callable[[], Any],
    issue_read: Callable[[Any, int], Any],
    make_writer: Callable[[], Any],
    issue_write: Callable[[Any, int], Any],
    close_reader: Callable[[Any], None] | None = None,
    close_writer: Callable[[Any], None] | None = None,
) -> ContentionOutcome:
    """Mede cada lado sozinho e os dois juntos, na mesma sessão.

    A ORDEM É DELIBERADA: os dois isolados primeiro, o concorrente depois. Medir o concorrente antes
    deixaria o cache aquecido pelo dobro do trabalho quando a linha de base fosse tomada, e a razão
    resultante subestimaria a contenção — o erro cairia para o lado de dizer que está tudo bem.

    A linha de base sai daqui e de lugar nenhum: não há parâmetro para injetá-la. Comparar contra
    outra corrida é o que o [[B-060]] e o [[B-063]] documentam como erro, e a assinatura torna isso
    impossível em vez de proibi-lo por escrito.
    """
    leitura_isolada = run_load(
        make_reader, issue_read, spec.read_ops, spec.readers, close_client=close_reader
    )
    escrita_isolada = run_load(
        make_writer, issue_write, spec.write_ops, spec.writers, close_client=close_writer
    )

    # As duas cargas em paralelo. Cada `run_load` já gerencia o próprio pool; o que falta é
    # largá-los
    # ao mesmo tempo, e uma thread por lado basta — o paralelismo interno de cada carga é do pool
    # dela, não desta camada.
    resultados: dict[str, LoadResult] = {}

    def _correr(nome: str, fn: Callable[[], LoadResult]) -> None:
        resultados[nome] = fn()

    fios = [
        threading.Thread(
            target=_correr,
            args=(
                "read",
                lambda: run_load(
                    make_reader, issue_read, spec.read_ops, spec.readers, close_client=close_reader
                ),
            ),
        ),
        threading.Thread(
            target=_correr,
            args=(
                "write",
                lambda: run_load(
                    make_writer,
                    issue_write,
                    spec.write_ops,
                    spec.writers,
                    close_client=close_writer,
                ),
            ),
        ),
    ]
    for fio in fios:
        fio.start()
    for fio in fios:
        fio.join()

    # Um lado sem NENHUM sucesso nao produz percentil, e uma razao contra `None` sai `None` — que
    # se le como "sem contencao" e significa "nada rodou". Encontrado rodando o comando contra um
    # adapter que nao implementa as operacoes: todas as tentativas erraram, `run_load` as registrou
    # (corretamente), e o relatorio saiu cheio de `null` sem dizer por que.
    #
    # Recusar aqui e a mesma disciplina que o `theodb.explain_scan` do produto aplica quando o
    # indice nao responde ao operador pedido: e melhor nao responder do que responder o vazio.
    for nome, isolado, sob in (
        ("leitura", leitura_isolada, resultados["read"]),
        ("escrita", escrita_isolada, resultados["write"]),
    ):
        if isolado.successes == 0 or sob.successes == 0:
            raise MeasurementError(
                f"o lado de {nome} nao completou nenhuma operação com sucesso "
                f"(isolado: {isolado.successes}/{isolado.errors + isolado.successes}, "
                f"concorrente: {sob.successes}/{sob.errors + sob.successes}). "
                "Percentis sobre zero sucessos saem `None`, e uma razão contra `None` se lê como "
                "'sem contenção' quando significa 'nada rodou' — o adapter provavelmente não "
                "implementa a operação pedida.",
                context=ErrorContext(phase=Phase.MEASUREMENT),
            )

    return ContentionOutcome(
        read=SideOutcome(isolated=leitura_isolada, concurrent=resultados["read"]),
        write=SideOutcome(isolated=escrita_isolada, concurrent=resultados["write"]),
        regime=spec.regime,
    )
