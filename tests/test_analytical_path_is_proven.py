"""B-058 bullet 4 — medir heap sob o rotulo do colunar e o defeito que este portao impede.

`assert_analytical_path` existia no `PostgresAdapter` (prova por catalogo E por plano) e o
`AlloyDbOmniAdapter` chegou a estende-lo com tres fatos separados — engine ligado, store populado,
planner preferindo colunar. **Nada em `src/bench/` o chamava.** O unico chamador no repositorio era
um `super()` dentro do proprio override.

E o mesmo defeito que o B-063 documenta sobre o `assert_index_used`: escrito, correto, sem chamador,
com o docstring anunciando o invariante como em vigor.

O B-058 registra o custo real da omissao: um avaliador terceiro perdeu uma corrida INTEIRA porque o
store colunar ficava vazio em silencio — todo plano caia para heap, sem erro e sem aviso.
"""

from __future__ import annotations

from theodb_bench.adapters.base import (
    AnalyticalQuery,
    AnalyticalResult,
    AnalyticalTable,
    SystemAdapter,
)
from theodb_bench.adapters.fake import FakeAdapter
from theodb_bench.errors import AdapterError, ErrorContext, Phase


class _AdapterQueCaiParaHeap(FakeAdapter):
    """Um motor que armazena colunar e PLANEJA heap — o modo de falha silencioso.

    Estende o `FakeAdapter` em vez de reimplementar a superficie abstrata inteira: o que este
    teste mede e o PORTAO, e escrever nove metodos vazios so para instanciar poria ruido entre o
    leitor e a propriedade.
    """

    system_id = "cai-para-heap"

    def __init__(self, linhas: int = 10) -> None:
        super().__init__()
        self.consultas_cronometradas = 0
        self.linhas = linhas

    def capabilities(self) -> dict[str, bool]:
        return {"analytical": True, "columnar": True, "parquet": True}

    def supports(self, capability: str) -> bool:
        return self.capabilities().get(capability, False)

    def assert_analytical_path(
        self, table: AnalyticalTable, query: AnalyticalQuery | None = None
    ) -> None:
        if table.path == "columnar":
            raise AdapterError(
                "o plano nao usou o caminho colunar: `theodb_columnar_agg` ausente",
                context=ErrorContext(phase=Phase.MEASUREMENT, system=self.system_id),
            )

    def execute_analytical(
        self, table: AnalyticalTable, query: AnalyticalQuery
    ) -> AnalyticalResult:
        # A resposta CERTA de proposito: o portao do oraculo ja invalida resposta errada, e um
        # duplo que errasse a conta faria este teste passar pelo motivo errado.
        self.consultas_cronometradas += 1
        return AnalyticalResult(rows=((self.linhas,),), wall_seconds=0.001)


def test_the_base_declares_the_proof_so_the_harness_can_ask_for_it() -> None:
    """Sem o metodo na base, o arnes nao tem como PEDIR a prova a um adapter qualquer.

    Era exatamente por isso que ela nunca era pedida: a implementacao existia no
    `PostgresAdapter`, e o caminho de medicao fala com `SystemAdapter`.
    """
    assert hasattr(SystemAdapter, "assert_analytical_path")


def test_the_default_is_honest_silence_not_an_exception() -> None:
    """Um motor sem planner nao tem caminho a conferir.

    `NotImplementedError` quebraria a substituibilidade num metodo que o arnes chama para TODO
    adapter registrado — pior que nao verificar.
    """

    tabela = AnalyticalTable(name="t", path="columnar", columns=("a",))
    # Direto na base: o default e dela, e e ele que garante a substituibilidade. O `mypy` sabe que
    # a assinatura devolve `None`, entao a asserção e sobre NAO LEVANTAR — que e a propriedade.
    SystemAdapter.assert_analytical_path(FakeAdapter(), tabela)


def test_a_path_that_falls_back_is_invalid_and_never_timed() -> None:
    """O coracao do portao: a medida e INVALIDADA **antes** de qualquer cronometragem.

    Um numero produzido por um plano que caiu para heap nao e um numero do colunar lento — e um
    numero do HEAP com o rotulo errado. Cronometra-lo primeiro e decidir depois deixaria o valor
    no artefato.
    """
    from theodb_bench.bench.analytical import AnalyticalBenchmark, AnalyticalWorkload

    workload = AnalyticalWorkload(
        row_count=10,
        paths=("row", "columnar"),
        queries=(AnalyticalQuery(id="total_rows", description="conta"),),
        repetitions=2,
        warmup_queries=1,
    )
    bench = AnalyticalBenchmark(workload)
    adapter = _AdapterQueCaiParaHeap()

    medida = bench.run_query(adapter, "columnar", workload.queries[0])

    assert medida.status == "invalid", "um plano que caiu para heap nao produz medida valida"
    assert "colunar" in (medida.status_detail or "")
    assert adapter.consultas_cronometradas == 0, (
        "a prova falhou, entao NADA pode ter sido cronometrado — nem o aquecimento"
    )


def test_a_path_that_proves_itself_is_measured_normally() -> None:
    """O contrapositivo: o portao nao pode reprovar o caminho que esta correto."""
    from theodb_bench.bench.analytical import AnalyticalBenchmark, AnalyticalWorkload

    workload = AnalyticalWorkload(
        row_count=10,
        paths=("row",),
        queries=(AnalyticalQuery(id="total_rows", description="conta"),),
        repetitions=2,
        warmup_queries=1,
    )
    bench = AnalyticalBenchmark(workload)
    adapter = _AdapterQueCaiParaHeap()

    medida = bench.run_query(adapter, "row", workload.queries[0])

    assert medida.status != "invalid", "o caminho `row` prova-se e tem de ser medido"
    assert adapter.consultas_cronometradas > 0
