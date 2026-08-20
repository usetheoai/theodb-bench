"""B-063 — o arnês passou a VERIFICAR o caminho de acesso, e não só a nomeá-lo.

Medido em 2026-08-17 no droplet `138.197.22.192`: `assert_index_used` não tinha chamador
nenhum (`grep -rn` devolvia a definição e dois comentários), estava quebrado se chamado
(dois placeholders, um parâmetro), e `SET enable_seqscan = off` aparecia UMA vez no
repositório — dentro de um docstring, em nenhum código executável. O docstring do módulo
anunciava o invariante I5, *"The index is forced **and** verified"*, e ele era falso nas
duas metades.

O que estes testes travam, e é a razão de existirem: o método voltar a ser código morto.
Um exemplar citado como "a disciplina exata" por outro item, que nunca roda, é pior que a
ausência dele — ensina o padrão errado com a autoridade de estar no repositório.

LIMITE DECLARADO: nada aqui foi exercitado contra servidor vivo. O droplet do B-059 foi
destruído e não há substituto ([[B-073]], [[B-075]]). A verificação de plano é coberta por
unidade nas duas pontas — o hook da base e a implementação Postgres — e a primeira corrida
real contra um servidor é o que fecha a lacuna.
"""

from __future__ import annotations

import numpy as np
import pytest
from theodb_bench.adapters.base import KnnQuery, KnnResult, SystemAdapter
from theodb_bench.adapters.fake import FakeAdapter
from theodb_bench.adapters.postgres import PgvectorAdapter
from theodb_bench.bench.vector import VectorBenchmark, VectorWorkload


def _workload() -> VectorWorkload:
    return VectorWorkload(corpus_size=64, dimension=8, query_count=8, k=4, warmup_queries=2)


class _SpyAdapter(FakeAdapter):
    """Conta as verificações pedidas, e responde às consultas por si.

    O `execute` do `FakeAdapter` exige o estado de prontidão do sistema simulado — máquina
    legítima, e que não é o assunto aqui. Responder direto mantém o teste sobre UM
    comportamento (`rules/testing.md` § 3) em vez de sobre o setup.
    """

    def __init__(self) -> None:
        super().__init__()
        self.verifications: list[KnnQuery] = []

    def verify_access_path(self, query: KnnQuery) -> None:
        self.verifications.append(query)

    def execute(self, query: KnnQuery) -> KnnResult:
        return KnnResult(ids=(0,), distances=(0.0,), latency_seconds=0.0)


def test_warm_up_asks_the_adapter_to_verify_the_access_path() -> None:
    """O chamador que faltava. Removê-lo faz este teste falhar — que é o bullet 1 do B-063.

    Na janela NÃO cronometrada de propósito: um EXPLAIN por consulta medida acrescentaria
    uma ida ao servidor dentro do relógio e o número passaria a descrever o arnês.
    """
    bench = VectorBenchmark(_workload())
    adapter = _SpyAdapter()
    bench.warm_up(adapter)
    assert len(adapter.verifications) == 1, (
        "o caminho de medição deve pedir a verificação do plano exatamente uma vez, "
        "antes da janela cronometrada"
    )


def test_verification_happens_before_any_warm_up_query() -> None:
    """Verificar depois mediria um plano que já influenciou o cache."""
    ordem: list[str] = []

    class _Ordered(_SpyAdapter):
        def verify_access_path(self, query: KnnQuery) -> None:
            ordem.append("verify")

        def execute(self, query: KnnQuery) -> KnnResult:
            ordem.append("query")
            return super().execute(query)

    VectorBenchmark(_workload()).warm_up(_Ordered())
    assert ordem[0] == "verify", f"a verificação deve vir primeiro, veio {ordem[:3]}"


def test_the_base_adapter_verifies_nothing_and_does_not_raise() -> None:
    """LSP: um sistema sem planner não tem caminho de acesso a conferir.

    O default é silêncio honesto e NÃO `NotImplementedError` — um subtipo que explode onde
    o supertipo é chamado quebra a substituibilidade, e o arnês chama isto para todo
    adapter registrado.
    """
    query = KnnQuery(table="bench_vectors", vector=np.zeros(8, dtype=np.float32), k=4)
    SystemAdapter.verify_access_path(FakeAdapter(), query)  # não levanta


def test_postgres_verifies_every_index_it_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """A implementação concreta delega ao `assert_index_used` — o método que era morto."""
    adapter = PgvectorAdapter()
    adapter._built_indexes.add("bench_hnsw")
    pedidos: list[str] = []
    monkeypatch.setattr(adapter, "assert_index_used", lambda q, name: pedidos.append(name))
    adapter.verify_access_path(
        KnnQuery(table="bench_vectors", vector=np.zeros(8, dtype=np.float32), k=4)
    )
    assert pedidos == ["bench_hnsw"]


def test_postgres_with_no_index_built_verifies_nothing() -> None:
    """Uma corrida de busca exata não construiu índice, e exigir um seria inventar defeito."""
    adapter = PgvectorAdapter()
    adapter._built_indexes.clear()
    adapter.verify_access_path(
        KnnQuery(table="bench_vectors", vector=np.zeros(8, dtype=np.float32), k=4)
    )  # não levanta, não consulta o servidor
