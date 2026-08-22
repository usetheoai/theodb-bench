"""Os clientes da contencao tem de medir a MESMA configuracao que a suite analitica.

MEDIDO em 2026-08-22. O `theodb-bench contention` reportou p50 de 15,7 s para `count(*)` numa tabela
colunar de 40M linhas. A suite analitica, no mesmo caminho, reporta 4,8 ms a 200k. A diferenca nao era
do volume nem do gerador de carga: era de CONFIGURACAO.

`_apply_analytical_session(table)` — que liga `theodb.enable_columnar_agg` — e chamado apenas dentro de
`load_analytical` (`adapters/postgres.py:944`), na conexao que CARREGA a tabela. E um ajuste de SESSAO.
A suite analitica carrega e mede na mesma conexao, entao herda o ajuste. A contencao cria clientes
novos, que nunca passam pelo carregador — e mede o colunar SEM o pushdown.

O comentario do proprio adapter descreve o defeito antes de ele acontecer: *"uma GUC que vem desligada
e silenciosamente continua desligada e como uma medicao acaba descrevendo uma configuracao que ninguem
rodou"*.
"""

from __future__ import annotations

from theodb_bench.adapters.base import AnalyticalTable
from theodb_bench.cli import _contention_client_factory


class _AdapterEspiao:
    def __init__(self) -> None:
        self.chamadas: list[str] = []

    def prepare(self) -> None:
        self.chamadas.append("prepare")

    def start(self) -> None:
        self.chamadas.append("start")

    def wait_ready(self) -> None:
        self.chamadas.append("wait_ready")

    def _apply_analytical_session(self, table: AnalyticalTable) -> None:
        self.chamadas.append(f"sessao:{table.path}")


def _tabela() -> AnalyticalTable:
    return AnalyticalTable(name="bench_contention", columns=("id", "value"), path="columnar")


def test_o_cliente_recebe_a_sessao_analitica_do_caminho() -> None:
    espiao = _AdapterEspiao()
    fabrica = _contention_client_factory(lambda: espiao, _tabela())

    fabrica()

    assert espiao.chamadas == ["prepare", "start", "wait_ready", "sessao:columnar"], (
        "sem a sessao analitica o cliente mede o colunar SEM pushdown — outra configuracao, "
        f"nao outra medida: {espiao.chamadas}"
    )


def test_a_sessao_vem_DEPOIS_de_conectar() -> None:
    """`SET` precisa de conexao. Aplicar antes do `start` levantaria, e o cliente cairia fora com o
    erro engolido pelo contador de operacoes — que foi como este defeito se escondeu."""
    espiao = _AdapterEspiao()
    _contention_client_factory(lambda: espiao, _tabela())()

    assert espiao.chamadas.index("start") < espiao.chamadas.index("sessao:columnar")


def test_um_adapter_sem_sessao_analitica_nao_quebra() -> None:
    """Nem todo adapter implementa o gancho; a fabrica nao pode exigir que implemente."""

    class _Simples:
        def __init__(self) -> None:
            self.chamadas: list[str] = []

        def prepare(self) -> None:
            self.chamadas.append("prepare")

        def start(self) -> None:
            self.chamadas.append("start")

        def wait_ready(self) -> None:
            self.chamadas.append("wait_ready")

    s = _Simples()
    _contention_client_factory(lambda: s, _tabela())()
    assert s.chamadas == ["prepare", "start", "wait_ready"]
