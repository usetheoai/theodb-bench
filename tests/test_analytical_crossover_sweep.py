"""B-058 bullet 2 — "abaixo de quantas linhas o colunar perde" precisa de uma CURVA.

O item registra o crossover do concorrente (o avaliador do AlloyDB mediu a inversao em algumas
centenas de milhares) e diz, sobre o nosso: nunca medido, so estimado. Um N unico nao responde
"abaixo de quantas" — responde "neste N", que e outra pergunta.
"""

from __future__ import annotations

from theodb_bench.adapters.base import AnalyticalQuery, AnalyticalResult, AnalyticalTable
from theodb_bench.adapters.fake import FakeAdapter
from theodb_bench.bench.analytical import AnalyticalBenchmark, AnalyticalWorkload


class _Contador(FakeAdapter):
    """Registra em que contagem de linhas cada carga e cada consulta aconteceu."""

    system_id = "contador"

    def __init__(self) -> None:
        super().__init__()
        self.cargas: list[int] = []
        self.consultas: list[int] = []
        self.linhas_atuais = 0
        # O ciclo de vida e contrato do adapter, e ele o cobra: carregar sem `start` levanta
        # `SystemUnavailableError`. Cumpri-lo aqui e o teste medir o que se propoe.
        self.prepare()
        self.start()
        self.wait_ready()

    def capabilities(self) -> dict[str, bool]:
        return {"analytical": True, "columnar": True, "parquet": True}

    def supports(self, capability: str) -> bool:
        return self.capabilities().get(capability, False)

    def load_analytical(self, table: AnalyticalTable, rows):  # type: ignore[no-untyped-def]
        linhas = list(rows)
        self.linhas_atuais = len(linhas)
        self.cargas.append(len(linhas))
        return super().load_analytical(table, linhas)

    def execute_analytical(
        self, table: AnalyticalTable, query: AnalyticalQuery
    ) -> AnalyticalResult:
        self.consultas.append(self.linhas_atuais)
        return AnalyticalResult(rows=((self.linhas_atuais,),), wall_seconds=0.001)


def _workload(sweep: tuple[int, ...]) -> AnalyticalWorkload:
    return AnalyticalWorkload(
        row_count=sweep[0] if sweep else 10,
        row_count_sweep=sweep,
        paths=("row",),
        queries=(AnalyticalQuery(id="total_rows", description="conta"),),
        repetitions=1,
        warmup_queries=0,
    )


def test_an_empty_sweep_keeps_the_single_row_count_behaviour() -> None:
    """A propriedade que as suites existentes protegem: sem varredura, nada muda."""
    bench = AnalyticalBenchmark(_workload(()))
    adapter = _Contador()
    bench.load(adapter)
    pontos = bench.points(adapter, repetitions=1)
    assert [p.parameters["row_count"] for p in pontos] == [10]
    assert adapter.cargas == [10], "uma carga so, no N declarado"


def test_the_sweep_reloads_the_data_at_every_row_count() -> None:
    """Cada N tem dados PROPRIOS. Medir o N grande sobre a tabela do N pequeno mede outra coisa."""
    bench = AnalyticalBenchmark(_workload((10, 20, 40)))
    adapter = _Contador()
    pontos = bench.points(adapter, repetitions=1)

    assert adapter.cargas == [10, 20, 40], "uma carga por contagem, na ordem da varredura"
    assert sorted(p.parameters["row_count"] for p in pontos) == [10, 20, 40]
    assert adapter.consultas == [10, 20, 40], (
        "cada consulta rodou sobre a tabela do SEU N — se um N medisse sobre os dados de outro, "
        "a curva do crossover seria de outra coisa"
    )


def test_the_oracle_is_rebuilt_per_row_count() -> None:
    """O oraculo e a resposta certa PARA ESTE corpus.

    Reusar o de outro N invalidaria toda medida da varredura por resposta errada — ou, pior, uma
    coincidencia passaria e a curva sairia de dados que ninguem conferiu.
    """
    bench = AnalyticalBenchmark(_workload((10, 20, 40)))
    adapter = _Contador()
    pontos = bench.points(adapter, repetitions=1)
    assert all(p.status == "measured" for p in pontos), (
        f"o oraculo nao acompanhou o N: {[(p.label, p.status, p.status_detail) for p in pontos]}"
    )


def test_the_sweep_does_not_load_before_points_owns_it() -> None:
    """`load()` do orquestrador nao pode gastar uma carga que o `points()` vai sobrescrever."""
    bench = AnalyticalBenchmark(_workload((10, 20)))
    adapter = _Contador()
    assert bench.load(adapter) is None
    assert adapter.cargas == [], "numa varredura, quem carrega e o points()"


def test_the_sweep_is_declared_in_the_artifact() -> None:
    """Quem le o artefato precisa saber que houve varredura, e de quais valores."""
    payload = _workload((10, 20, 40)).benchmark_payload()
    assert payload["parameters"]["row_count"] == [10, 20, 40]
    assert payload["workload"]["operation_count"] == 3, "1 consulta x 1 caminho x 3 contagens"


def test_a_single_row_count_does_not_claim_a_sweep() -> None:
    """Repetir o `row_count` como parametro VARRIDO sugeriria varredura onde houve um ponto so."""
    payload = _workload(()).benchmark_payload()
    assert "row_count" not in payload["parameters"]


def test_the_data_is_loaded_once_per_row_count_not_once_per_repetition() -> None:
    """MEDIDO em 2026-08-21: eram QUATRO cargas por caminho numa corrida de 3 repeticoes.

    O `points()` chamava `run()` uma vez por repeticao e cada chamada recarregava todos os
    caminhos; somando a carga do orquestrador, dava 1 + 3. A 2M linhas na varredura do crossover
    isso e a maior parte do custo da corrida, gasto para reescrever a MESMA tabela.

    O portao aqui e economico e nao estetico: uma corrida que gasta 4x o necessario e uma corrida
    que alguem deixa de rodar.
    """
    bench = AnalyticalBenchmark(_workload((10, 20)))
    adapter = _Contador()
    bench.points(adapter, repetitions=3)
    assert adapter.cargas == [10, 20], (
        f"uma carga por contagem, nao por repeticao — foram {adapter.cargas}"
    )


def test_running_standalone_still_loads() -> None:
    """`run()` sozinho continua carregando: quatro testes dependem disso, e o contrato e dele."""
    bench = AnalyticalBenchmark(_workload(()))
    adapter = _Contador()
    bench.run(adapter)
    assert adapter.cargas == [10], "quem chama run() sozinho nao carregou antes"


def test_a_sweep_run_does_not_crash_the_runner_when_load_returns_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """`Benchmark.load` declara `float | None`, e o runner formatava o `None` DIRETO.

    MEDIDO no droplet: a corrida da varredura abortava com
    `TypeError: unsupported format string passed to NoneType.__format__` — e o relatorio
    culpava o SISTEMA SOB TESTE (`sut_alive: FAIL`) por um defeito do arnes. O contrato ja
    dizia "segundos, ou None quando nada foi carregado"; quem nao honrava era o runner.

    Este teste passa pelo `run_benchmark` inteiro, e nao por uma copia da expressao — uma
    asserção sobre a propria copia nao provaria nada sobre o runner.
    """
    from theodb_bench.adapters.fake import FakeAdapter
    from theodb_bench.runner import RunRequest, run_benchmark

    workload = AnalyticalWorkload(
        row_count=10,
        row_count_sweep=(10, 20),
        paths=("row",),
        queries=(AnalyticalQuery(id="total_rows", description="conta"),),
        repetitions=1,
        warmup_queries=0,
    )
    resultado = run_benchmark(
        RunRequest(
            benchmark_id="analytical/crossover/row-count",
            workload=workload,
            adapter_factory=FakeAdapter,
            results_root=tmp_path / "resultados",
            repetitions=1,
        )
    )
    carga = (resultado.bundle.root / "raw" / "load.log").read_text(encoding="utf-8")
    assert "load_seconds=none" in carga, (
        "o `None` tem de ser REGISTRADO: um `load.log` ausente se leria como carga de custo zero"
    )


def test_an_invalid_measurement_reaches_the_artifact_as_failed() -> None:
    """`invalid` e vocabulario INTERNO; o artefato conhece `measured|unsupported|skipped|failed`.

    MEDIDO no droplet: a corrida completou as 72 medidas e **o artefato foi recusado na validacao**
    — `$.points.6.status: 'invalid' is not one of [...]`, seis vezes, uma por contagem de linhas.

    O defeito e ANTERIOR ao portao de caminho: a resposta errada do oraculo ja produzia `invalid`
    desde sempre. Nunca disparou porque o oraculo nunca falhou nas suites — o que e outro caso de
    portao que so passa porque nunca foi exercitado.

    Nao se acrescenta um quinto termo: ele sobreporia `failed`, e a distincao que importa (recusada
    contra caiu) vive no `status_detail`.
    """
    from theodb_bench.adapters.base import AnalyticalResult

    class _RespostaErrada(_Contador):
        def execute_analytical(
            self, table: AnalyticalTable, query: AnalyticalQuery
        ) -> AnalyticalResult:
            self.consultas.append(self.linhas_atuais)
            return AnalyticalResult(rows=((-1,),), wall_seconds=0.001)

    bench = AnalyticalBenchmark(_workload(()))
    pontos = bench.points(_RespostaErrada(), repetitions=1)

    assert [p.status for p in pontos] == ["failed"]
    assert "oracle" in (pontos[0].status_detail or "").lower(), (
        "a razao tem de sobreviver ao mapeamento — e ela que distingue recusada de caida"
    )
